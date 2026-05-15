"""Stochastic-decoder variant of LSTM_VAE.

Tests the hypothesis raised by the 3-lens diagnostics in v1 (PROJECT.md §3.9):
the autoregressive *deterministic* decoder regresses to the mean and discards
high-frequency content. If we inject Gaussian noise into the decoder's hidden
state at each timestep, the decoder is forced to model the full conditional
distribution rather than its mean. If this closes the λ₁ and DET gap with real
data, that is a causal demonstration that the diagnostics correctly identified
the failure mode.

Implementation note: this is the minimal-change variant. Only the decoder is
stochastic; the encoder, bottleneck, and max-value-predictor heads are
unchanged. The forward / generate / decode methods are inherited from LSTM_VAE
and overridden where they call into the decoder autoregressive loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cvae import LSTM_VAE


class StochasticLSTMVAE(LSTM_VAE):
    """LSTM-VAE with Gaussian noise injection in the decoder hidden state."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Decoder-noise scale: learned scalar, initialized to the value in config
        # (default 0.05). Clamped >= 0 via softplus so it can't go negative.
        init = float(config.get("decoder_noise_init", 0.05))
        # We parameterize through softplus so the learned param is unconstrained.
        # softplus^{-1}(init) for the inverse.
        import math
        raw_init = math.log(math.expm1(max(1e-4, init)))
        self.decoder_noise_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        print(f"Stochastic decoder noise initialized at σ ≈ {init:.4f} (learnable)")

    @property
    def decoder_noise_sigma(self) -> torch.Tensor:
        return nn.functional.softplus(self.decoder_noise_raw)

    def _inject_noise(self, hidden_state: tuple) -> tuple:
        """Add Gaussian noise to the decoder hidden state (h, c).

        Only h is perturbed (c is the cell state which already carries gating;
        perturbing both is overkill and tends to destabilize training in early
        experiments). The noise scale is the learned σ.
        """
        h, c = hidden_state
        sigma = self.decoder_noise_sigma
        if self.training:
            h = h + sigma * torch.randn_like(h)
        else:
            # At eval/generation time we still inject noise — that is the whole
            # point. The diagnostics are run on generation-time samples; if we
            # disabled noise here, the protocol would not see what the user gets.
            h = h + sigma * torch.randn_like(h)
        return (h, c)

    def forward(self, X: torch.Tensor, max_vals: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> tuple:
        """Forward pass with noise-injected decoder.

        Re-implements the relevant chunk of the parent forward() because the
        decoder loop is inlined there. Encoder + bottleneck + max-value-head
        logic is copied verbatim from LSTM_VAE.forward to keep training-time
        behavior identical except for the decoder loop.
        """
        batch_size = X.size(0)
        X_rnn = X.permute(0, 2, 1)

        # --- Encoding (unchanged) ---
        _, (h_n, _) = self.encoder_rnn(X_rnn)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)

        if self.use_scale_conditioning:
            if max_vals is None:
                raise ValueError("max_vals must be provided when use_scale_conditioning=True")
            z_shape = self.shape_projector(encoded_summary)
            z_scale = self.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
        else:
            bottleneck_input = encoded_summary

        mu = self.fc_mu(bottleneck_input)
        log_var = self.fc_log_var(bottleneck_input)
        z = self.sample(mu, log_var)

        # --- Max-value head (unchanged) ---
        max_vals_pred_transformed_partial = self.max_value_predictor(z)
        max_vals_pred_partial = self.inverse_transform_max_values(max_vals_pred_transformed_partial)
        ones = torch.ones(batch_size, 1, device=z.device)
        max_vals_pred = torch.cat([ones, max_vals_pred_partial], dim=1)
        ones_transformed = self.transform_max_values(ones)
        max_vals_pred_transformed = torch.cat([ones_transformed, max_vals_pred_transformed_partial], dim=1)

        # --- Decoder: same as parent but with noise injection at every step ---
        h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)

        current_curve_input = torch.zeros(batch_size, 1, self.n_curves, device=X.device)
        outputs = []
        z_step_input = z.unsqueeze(1)

        for t in range(self.seq_len):
            # *** NEW *** inject noise into hidden state before each step
            hidden_state = self._inject_noise(hidden_state)

            step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            output = self.output_map(output)
            outputs.append(output)

            use_teacher_force = torch.rand(1) < teacher_forcing_ratio
            if self.training and use_teacher_force:
                current_curve_input = X_rnn[:, t, :].unsqueeze(1)
            else:
                current_curve_input = output

        X_hat_rnn = torch.cat(outputs, dim=1)
        X_hat = X_hat_rnn.permute(0, 2, 1)
        X_hat = torch.clamp(X_hat, min=0, max=1)

        return X_hat, mu, log_var, max_vals_pred, max_vals_pred_transformed

    def _decode_loop_stochastic(self, z: torch.Tensor) -> tuple:
        """Shared decoder loop for generate() and decode() — with noise."""
        num_samples = z.size(0)
        device = z.device

        predicted_max_vals_transformed_partial = self.max_value_predictor(z)
        predicted_max_vals_partial = self.inverse_transform_max_values(predicted_max_vals_transformed_partial)
        ones = torch.ones(num_samples, 1, device=device)
        predicted_max_vals = torch.cat([ones, predicted_max_vals_partial], dim=1)

        h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
        c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)

        current_curve_input = torch.zeros(num_samples, 1, self.n_curves, device=device)
        z_step_input = z.unsqueeze(1)

        generated_steps = []
        for _ in range(self.seq_len):
            hidden_state = self._inject_noise(hidden_state)
            step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            output = self.output_map(output)
            current_curve_input = output
            generated_steps.append(output)

        generated_sequences_rnn = torch.cat(generated_steps, dim=1)
        generated_sequences = generated_sequences_rnn.permute(0, 2, 1)
        generated_sequences = torch.clamp(generated_sequences, min=0, max=1)
        return generated_sequences, predicted_max_vals

    def generate(self, num_samples: int, device: torch.device) -> tuple:
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self._decode_loop_stochastic(z)

    def decode(self, z: torch.Tensor) -> tuple:
        self.eval()
        with torch.no_grad():
            return self._decode_loop_stochastic(z)
