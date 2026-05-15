"""KAN-VAE for GLV trajectories (model 6 in the pivot lineup).

We keep the LSTM_VAE backbone — including the autoregressive recurrent decoder
— and replace the MLP heads (scale_encoder, max_value_predictor, output_map)
with KAN layers from the vendored ``efficient_kan`` reference implementation
(Liu et al. 2024, arXiv:2404.19756).

Rationale: this tests "KAN as a function approximator" while holding the
sequence-modeling component fixed. If KAN-VAE behaves materially differently
on the 3-lens metrics from LSTM_VAE, the difference is attributable to the
function-approximation basis. If it does not, the result is informative for
the "where does KAN help?" literature.

API is fully compatible with LSTM_VAE.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .efficient_kan import KANLinear


class _KANSeq(nn.Module):
    """Sequential composition of KANLinear layers with optional dropout."""

    def __init__(self, dims, dropout: float = 0.0):
        super().__init__()
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(KANLinear(in_d, out_d))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class KANVAE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        self.n_curves = config.get("n_curves", 7)
        self.seq_len = config.get("seq_len", 65)
        self.latent_dim = config["latent_dim"]
        self.max_value_dim = self.n_curves - 1
        self.scale_prediction_mode = config.get("scale_prediction_mode", "log")
        self.use_scale_conditioning = config.get("use_scale_conditioning", True)
        self.rnn_hidden_size = config.get("rnn_hidden_size", 256)
        self.rnn_num_layers = config.get("rnn_num_layers", 2)

        print(f"KAN-VAE Parameters: Latent Dim={self.latent_dim}")
        print(f"Scale Conditioning: {'ENABLED' if self.use_scale_conditioning else 'DISABLED'}")

        # --- Encoder (unchanged LSTM backbone) ---
        self.encoder_rnn = nn.LSTM(
            input_size=self.n_curves,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        encoder_output_dim = self.rnn_num_layers * 2 * self.rnn_hidden_size

        # --- Scale conditioning: KAN-based scale encoder ---
        if self.use_scale_conditioning:
            # Linear projector for the LSTM summary (KAN here is overkill;
            # we replace the *small* projection MLPs instead, which is where
            # KAN's literature has shown it helps most).
            self.shape_projector = nn.Linear(encoder_output_dim, self.latent_dim)
            # KAN replaces the 2-layer MLP scale encoder
            self.scale_encoder = _KANSeq([self.n_curves, self.latent_dim, self.latent_dim])
            bottleneck_input_dim = self.latent_dim * 2
        else:
            bottleneck_input_dim = encoder_output_dim

        self.fc_mu = nn.Linear(bottleneck_input_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(bottleneck_input_dim, self.latent_dim)

        # --- Max-value head: KAN-based ---
        self.max_value_predictor = _KANSeq(
            [self.latent_dim, self.latent_dim // 2, self.max_value_dim],
            dropout=0.2,
        )

        # --- Decoder ---
        decoder_initial_state_dim = self.rnn_num_layers * self.rnn_hidden_size
        self.latent_to_hidden = nn.Linear(self.latent_dim, decoder_initial_state_dim)
        self.latent_to_cell = nn.Linear(self.latent_dim, decoder_initial_state_dim)

        decoder_input_dim = self.n_curves + self.latent_dim
        self.decoder_rnn = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )

        # Output head: KAN replaces the final Linear(rnn_hidden→n_curves).
        # NOTE: at full feature resolution this is the most expensive KAN swap;
        # we keep it because it's the closest to the data and where KAN's
        # expressivity is most likely to show.
        self.output_map = KANLinear(self.rnn_hidden_size, self.n_curves)

    # --- max-value transforms (mirror LSTM_VAE) ---
    def transform_max_values(self, max_vals):
        if self.scale_prediction_mode == "log":
            return torch.log(max_vals + 1e-8)
        elif self.scale_prediction_mode == "exp":
            return torch.exp(max_vals)
        return max_vals

    def inverse_transform_max_values(self, transformed_vals):
        if self.scale_prediction_mode == "log":
            return torch.exp(transformed_vals)
        elif self.scale_prediction_mode == "exp":
            return torch.log(torch.clamp(transformed_vals, min=1e-8))
        return transformed_vals

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X, max_vals=None, teacher_forcing_ratio=0.5):
        batch_size = X.size(0)
        X_rnn = X.permute(0, 2, 1)

        _, (h_n, _) = self.encoder_rnn(X_rnn)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)

        if self.use_scale_conditioning:
            if max_vals is None:
                raise ValueError("max_vals required when use_scale_conditioning=True")
            z_shape = self.shape_projector(encoded_summary)
            z_scale = self.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
        else:
            bottleneck_input = encoded_summary

        mu = self.fc_mu(bottleneck_input)
        log_var = self.fc_log_var(bottleneck_input)
        z = self.sample(mu, log_var)

        # Max-value head
        mvtp = self.max_value_predictor(z)
        mvp = self.inverse_transform_max_values(mvtp)
        ones = torch.ones(batch_size, 1, device=z.device)
        max_vals_pred = torch.cat([ones, mvp], dim=1)
        ones_t = self.transform_max_values(ones)
        max_vals_pred_t = torch.cat([ones_t, mvtp], dim=1)

        # Decoder
        h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, batch_size, self.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)

        current_curve_input = torch.zeros(batch_size, 1, self.n_curves, device=X.device)
        outputs = []
        z_step_input = z.unsqueeze(1)

        for t in range(self.seq_len):
            step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            # KAN output map. KANLinear expects (N, in_features), so we apply
            # per-step on the last dim. Output is (N, 1, n_curves).
            output_flat = output.reshape(-1, self.rnn_hidden_size)
            output_proj = self.output_map(output_flat)
            output = output_proj.view(batch_size, 1, self.n_curves)
            outputs.append(output)
            use_teacher_force = torch.rand(1) < teacher_forcing_ratio
            if self.training and use_teacher_force:
                current_curve_input = X_rnn[:, t, :].unsqueeze(1)
            else:
                current_curve_input = output

        X_hat_rnn = torch.cat(outputs, dim=1)
        X_hat = X_hat_rnn.permute(0, 2, 1)
        X_hat = torch.clamp(X_hat, 0, 1)
        return X_hat, mu, log_var, max_vals_pred, max_vals_pred_t

    def _decode_loop(self, z):
        num_samples = z.size(0)
        device = z.device
        mvtp = self.max_value_predictor(z)
        mvp = self.inverse_transform_max_values(mvtp)
        ones = torch.ones(num_samples, 1, device=device)
        max_vals_pred = torch.cat([ones, mvp], dim=1)

        h_0 = self.latent_to_hidden(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
        c_0 = self.latent_to_cell(z).view(self.rnn_num_layers, num_samples, self.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)

        current_curve_input = torch.zeros(num_samples, 1, self.n_curves, device=device)
        z_step_input = z.unsqueeze(1)

        generated_steps = []
        for _ in range(self.seq_len):
            step_input = torch.cat([current_curve_input, z_step_input], dim=-1)
            output, hidden_state = self.decoder_rnn(step_input, hidden_state)
            output_flat = output.reshape(-1, self.rnn_hidden_size)
            output_proj = self.output_map(output_flat)
            output = output_proj.view(num_samples, 1, self.n_curves)
            current_curve_input = output
            generated_steps.append(output)

        X_hat = torch.cat(generated_steps, dim=1).permute(0, 2, 1)
        X_hat = torch.clamp(X_hat, 0, 1)
        return X_hat, max_vals_pred

    def generate(self, num_samples, device):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self._decode_loop(z)

    def decode(self, z):
        self.eval()
        with torch.no_grad():
            return self._decode_loop(z)
