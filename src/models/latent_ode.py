"""Latent-ODE generative model for GLV trajectories (model 4 in the pivot lineup).

Architecture follows Rubanova et al. 2019 (Latent ODEs for Irregularly-Sampled
Time Series, NeurIPS 2019) adapted to our regular-time setup:

  Encoder:    BiLSTM over (7, 65)   →   (μ_0, log σ²_0) ∈ ℝ^L  (posterior on z(t=0))
  Dynamics:   dz/dt = f_θ(z, t),  f_θ a small MLP
  Integrator: torchdiffeq.odeint over the 65 timesteps
  Decoder:    autoregressive LSTM on z(t) at each step → trajectory + max-value head

We keep the API surface — forward / generate / decode — compatible with LSTM_VAE
so the unified eval harness can call it the same way.

Crucially, the **scale-conditioning** mechanism from LSTM_VAE is preserved here:
we encode max_vals via the same scale_encoder into z_scale, concatenate with
z_shape, then sample z_0 from the bottleneck. This keeps the comparison
"architecture only" — the input information is identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint

ACTIVATION = nn.SiLU()


class _ODEFunc(nn.Module):
    """Velocity field f_θ(z, t) for the latent ODE."""

    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        # t is a scalar tensor; we don't use it explicitly (autonomous ODE)
        return self.net(z)


class LatentODE(nn.Module):
    """Latent-ODE generative model with scale conditioning."""

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

        # ODE-specific config
        self.ode_hidden = config.get("ode_hidden_size", 128)
        self.ode_depth = config.get("ode_depth", 2)
        self.ode_method = config.get("ode_method", "rk4")
        self.ode_step_size = config.get("ode_step_size", 0.25)

        print(f"LatentODE Parameters: Latent Dim={self.latent_dim}")
        print(f"Scale Conditioning: {'ENABLED' if self.use_scale_conditioning else 'DISABLED'}")
        print(f"  ODE: hidden={self.ode_hidden}, depth={self.ode_depth}, method={self.ode_method}")

        # --- Encoder (BiLSTM over the sequence) ---
        self.encoder_rnn = nn.LSTM(
            input_size=self.n_curves,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        encoder_output_dim = self.rnn_num_layers * 2 * self.rnn_hidden_size

        # --- Scale conditioning (same shape encoder + scale encoder as LSTM_VAE) ---
        if self.use_scale_conditioning:
            self.shape_projector = nn.Linear(encoder_output_dim, self.latent_dim)
            self.scale_encoder = nn.Sequential(
                nn.Linear(self.n_curves, self.latent_dim),
                ACTIVATION,
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            bottleneck_input_dim = self.latent_dim * 2
        else:
            bottleneck_input_dim = encoder_output_dim

        self.fc_mu = nn.Linear(bottleneck_input_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(bottleneck_input_dim, self.latent_dim)

        # --- Max-value head from z_0 ---
        self.max_value_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.Dropout(0.2),
            ACTIVATION,
            nn.Linear(self.latent_dim // 2, self.max_value_dim),
        )

        # --- Latent ODE ---
        self.ode_func = _ODEFunc(self.latent_dim, hidden=self.ode_hidden, depth=self.ode_depth)

        # --- Decoder: simple per-timestep MLP from z(t) → curve values ---
        # (Choice: MLP per-step rather than autoregressive RNN. The whole point
        # of using an ODE is that the dynamics live in latent space; the
        # decoder should just be a memoryless readout.)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.rnn_hidden_size),
            ACTIVATION,
            nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size),
            ACTIVATION,
            nn.Linear(self.rnn_hidden_size, self.n_curves),
        )

        # Integration time points (normalized to [0, 1])
        self.register_buffer("t_points", torch.linspace(0.0, 1.0, self.seq_len))

    # ----- max-value transforms (mirror LSTM_VAE) -----
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

    # ----- core paths -----
    def _encode(self, X, max_vals=None):
        """X: (N, 7, T) → (mu, log_var) of z_0."""
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

        mu = self.fc_mu(bottleneck_input)
        log_var = self.fc_log_var(bottleneck_input)
        return mu, log_var

    def _integrate(self, z_0):
        """z_0: (N, L) → z_traj: (T, N, L)."""
        opts = {"step_size": self.ode_step_size} if self.ode_method in {"rk4", "euler"} else {}
        z_traj = odeint(
            self.ode_func, z_0, self.t_points, method=self.ode_method, options=opts
        )
        return z_traj  # (T, N, L)

    def _predict_max_vals(self, z_0):
        batch_size = z_0.size(0)
        mvtp = self.max_value_predictor(z_0)
        mvp = self.inverse_transform_max_values(mvtp)
        ones = torch.ones(batch_size, 1, device=z_0.device)
        max_vals_pred = torch.cat([ones, mvp], dim=1)
        ones_t = self.transform_max_values(ones)
        max_vals_pred_t = torch.cat([ones_t, mvtp], dim=1)
        return max_vals_pred, max_vals_pred_t

    def forward(self, X, max_vals=None, teacher_forcing_ratio=0.5):
        """Returns (X_hat, mu, log_var, max_vals_pred, max_vals_pred_transformed)
        matching LSTM_VAE's signature so the trainer can be reused unchanged.
        teacher_forcing_ratio is unused for this model (no autoregression) but
        accepted for API compatibility.
        """
        mu, log_var = self._encode(X, max_vals)
        z_0 = self.sample(mu, log_var)

        max_vals_pred, max_vals_pred_t = self._predict_max_vals(z_0)

        z_traj = self._integrate(z_0)             # (T, N, L)
        z_traj = z_traj.permute(1, 0, 2)          # (N, T, L)
        outputs = self.decoder(z_traj)            # (N, T, 7)
        X_hat = outputs.permute(0, 2, 1)          # (N, 7, T)
        X_hat = torch.clamp(X_hat, min=0, max=1)

        return X_hat, mu, log_var, max_vals_pred, max_vals_pred_t

    def generate(self, num_samples, device):
        self.eval()
        with torch.no_grad():
            z_0 = torch.randn(num_samples, self.latent_dim, device=device)
            max_vals_pred, _ = self._predict_max_vals(z_0)
            z_traj = self._integrate(z_0).permute(1, 0, 2)
            X_hat = torch.clamp(self.decoder(z_traj).permute(0, 2, 1), 0, 1)
            return X_hat, max_vals_pred

    def decode(self, z):
        self.eval()
        with torch.no_grad():
            max_vals_pred, _ = self._predict_max_vals(z)
            z_traj = self._integrate(z).permute(1, 0, 2)
            X_hat = torch.clamp(self.decoder(z_traj).permute(0, 2, 1), 0, 1)
            return X_hat, max_vals_pred
