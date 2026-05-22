"""Heteroscedastic-decoder variant of LSTM_VAE — M8.

Architectural fix for the smoothness-attractor pathology (PROJECT.md §1.3.0.2).
Instead of post-hoc Gaussian noise (m3 StochasticLSTMVAE injects σ into the
hidden state; the σ-sweep in §1.3.0.1 adds σ to outputs at eval-time), the
decoder here predicts both μ and σ per timestep per curve, and is trained
under Gaussian NLL rather than MSE. At generate-time we sample from the
predicted N(μ, σ).

This is the architectural test of whether VAEs sit at the smoothness attractor
*because* of the homoscedastic Gaussian decoder assumption (equivalent to MSE).
If σ is allowed to be data-dependent and the loss rewards correct uncertainty,
the decoder should naturally produce trajectories whose per-timestep variance
matches the real observation noise (σ ≈ 0.01 lognormal from
`generate_family_FIXED.py:203`).

Output convention: forward() returns x_hat of shape (N, 2C, T) — μ stacked
with raw log_σ along the channel axis. vae_loss() detects this shape and
branches to Gaussian-NLL. All other call sites unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .cvae import LSTM_VAE


MIN_SIGMA = 1e-3  # floor so log σ stays finite + KS test stays well-defined


class HeteroscedasticLSTMVAE(LSTM_VAE):
    """LSTM-VAE with a per-(t,c) heteroscedastic Gaussian output head."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.output_map = nn.Linear(self.rnn_hidden_size, 2 * self.n_curves)
        with torch.no_grad():
            self.output_map.bias[self.n_curves:].fill_(math.log(math.expm1(0.05)))
        print(f"HeteroscedasticLSTMVAE: output head 2×{self.n_curves} (μ, raw_log_σ); σ_floor={MIN_SIGMA}")

    def _split_output(self, raw: torch.Tensor) -> tuple:
        """Split (N, T, 2C) raw output into (μ, σ) — σ via softplus + floor."""
        mu = raw[..., :self.n_curves]
        raw_sigma = raw[..., self.n_curves:]
        sigma = nn.functional.softplus(raw_sigma) + MIN_SIGMA
        return mu, sigma

    def forward(self, X: torch.Tensor, max_vals: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> tuple:
        batch_size = X.size(0)
        X_rnn = X.permute(0, 2, 1)

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

        mu_z = self.fc_mu(bottleneck_input)
        log_var_z = self.fc_log_var(bottleneck_input)
        z = self.sample(mu_z, log_var_z)

        max_vals_pred_transformed_partial = self.max_value_predictor(z)
        max_vals_pred_partial = self.inverse_transform_max_values(max_vals_pred_transformed_partial)
        ones = torch.ones(batch_size, 1, device=z.device)
        max_vals_pred = torch.cat([ones, max_vals_pred_partial], dim=1)
        ones_transformed = self.transform_max_values(ones)
        max_vals_pred_transformed = torch.cat([ones_transformed, max_vals_pred_transformed_partial], dim=1)

        h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)

        current_curve_input = torch.zeros(batch_size, 1, self.n_curves, device=X.device)
        mus, raw_sigmas = [], []
        z_step_input = z.unsqueeze(1)

        for t in range(self.seq_len):
            step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            raw = self.output_map(output)
            mu_t = raw[..., :self.n_curves]
            raw_sigma_t = raw[..., self.n_curves:]
            mus.append(mu_t)
            raw_sigmas.append(raw_sigma_t)

            use_teacher_force = torch.rand(1) < teacher_forcing_ratio
            if self.training and use_teacher_force:
                current_curve_input = X_rnn[:, t, :].unsqueeze(1)
            else:
                current_curve_input = mu_t  # feed μ forward (not a sample) for training stability

        mu_rnn = torch.cat(mus, dim=1)
        log_sigma_rnn = torch.cat(raw_sigmas, dim=1)
        X_mu = mu_rnn.permute(0, 2, 1)
        X_log_sigma = log_sigma_rnn.permute(0, 2, 1)
        X_mu = torch.clamp(X_mu, min=0, max=1)
        X_hat = torch.cat([X_mu, X_log_sigma], dim=1)  # (N, 2C, T)

        return X_hat, mu_z, log_var_z, max_vals_pred, max_vals_pred_transformed

    def _decode_loop_sample(self, z: torch.Tensor) -> tuple:
        """Generate-time loop: sample from N(μ_t, σ_t) at each step."""
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
            step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            raw = self.output_map(output)
            mu_t, sigma_t = self._split_output(raw)
            sample_t = mu_t + sigma_t * torch.randn_like(mu_t)
            sample_t = torch.clamp(sample_t, min=0, max=1)
            current_curve_input = sample_t
            generated_steps.append(sample_t)

        generated_sequences_rnn = torch.cat(generated_steps, dim=1)
        generated_sequences = generated_sequences_rnn.permute(0, 2, 1)
        return generated_sequences, predicted_max_vals

    def generate(self, num_samples: int, device: torch.device) -> tuple:
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self._decode_loop_sample(z)

    def decode(self, z: torch.Tensor) -> tuple:
        self.eval()
        with torch.no_grad():
            return self._decode_loop_sample(z)
