"""Transformer-VAE for GLV trajectories (model 5 in the pivot lineup).

Architecture:
  Encoder:    Linear(7→d) + sinusoidal positional encoding + 4-layer Transformer
              encoder over (T, d). Mean-pooling over the time axis → posterior
              (μ, log σ²) ∈ ℝ^L.
  Decoder:    Linear(L→d) + positional encoding + 4-layer Transformer decoder
              with causal mask. Cross-attention reads a fixed-length "memory"
              token derived from z. Output projection to (T, 7).

Scale conditioning matches LSTM_VAE: max_vals → MLP → z_scale; concat with the
pooled-shape vector at the bottleneck.

API compatibility: same forward / generate / decode signature as LSTM_VAE,
including the teacher_forcing_ratio argument (unused, kept for trainer
compatibility).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

ACTIVATION = nn.SiLU()


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (N, T, D)
        return x + self.pe[:, : x.size(1)]


class TransformerVAE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        self.n_curves = config.get("n_curves", 7)
        self.seq_len = config.get("seq_len", 65)
        self.latent_dim = config["latent_dim"]
        self.max_value_dim = self.n_curves - 1
        self.scale_prediction_mode = config.get("scale_prediction_mode", "log")
        self.use_scale_conditioning = config.get("use_scale_conditioning", True)

        self.d_model = config.get("d_model", 128)
        self.n_heads = config.get("n_heads", 4)
        self.n_enc_layers = config.get("n_enc_layers", 4)
        self.n_dec_layers = config.get("n_dec_layers", 4)
        self.ff_dim = config.get("ff_dim", 256)
        self.dropout = config.get("dropout", 0.1)

        print(f"TransformerVAE: d_model={self.d_model} heads={self.n_heads} "
              f"enc_layers={self.n_enc_layers} dec_layers={self.n_dec_layers}")
        print(f"Scale Conditioning: {'ENABLED' if self.use_scale_conditioning else 'DISABLED'}")

        # --- Encoder ---
        self.input_proj = nn.Linear(self.n_curves, self.d_model)
        self.pos_enc = _PositionalEncoding(self.d_model, max_len=max(self.seq_len, 256))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=self.ff_dim, dropout=self.dropout, batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.n_enc_layers)

        # --- Scale conditioning ---
        if self.use_scale_conditioning:
            self.shape_projector = nn.Linear(self.d_model, self.latent_dim)
            self.scale_encoder = nn.Sequential(
                nn.Linear(self.n_curves, self.latent_dim),
                ACTIVATION,
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            bottleneck_input_dim = self.latent_dim * 2
        else:
            self.shape_projector = nn.Identity()
            bottleneck_input_dim = self.d_model

        self.fc_mu = nn.Linear(bottleneck_input_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(bottleneck_input_dim, self.latent_dim)

        # --- Max-value head ---
        self.max_value_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.Dropout(0.2),
            ACTIVATION,
            nn.Linear(self.latent_dim // 2, self.max_value_dim),
        )

        # --- Decoder ---
        # We feed a learned query sequence of length seq_len; cross-attention
        # reads z projected into a single memory token (or T tokens via repeat).
        self.z_to_memory = nn.Linear(self.latent_dim, self.d_model)
        self.query_embed = nn.Parameter(torch.randn(self.seq_len, self.d_model) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=self.ff_dim, dropout=self.dropout, batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.n_dec_layers)
        self.output_map = nn.Linear(self.d_model, self.n_curves)

    # --- max-value transforms ---
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

    def _encode(self, X, max_vals=None):
        X_seq = X.permute(0, 2, 1)               # (N, T, 7)
        h = self.input_proj(X_seq)               # (N, T, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)                      # (N, T, d_model)
        pooled = h.mean(dim=1)                   # (N, d_model)

        if self.use_scale_conditioning:
            if max_vals is None:
                raise ValueError("max_vals required when use_scale_conditioning=True")
            z_shape = self.shape_projector(pooled)
            z_scale = self.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
        else:
            bottleneck_input = pooled

        mu = self.fc_mu(bottleneck_input)
        log_var = self.fc_log_var(bottleneck_input)
        return mu, log_var

    def _decode_from_z(self, z):
        N = z.size(0)
        memory = self.z_to_memory(z).unsqueeze(1)         # (N, 1, d_model)
        # Repeat memory across seq_len for richer cross-attention
        memory = memory.expand(-1, self.seq_len, -1)      # (N, T, d_model)

        query = self.query_embed.unsqueeze(0).expand(N, -1, -1)  # (N, T, d_model)
        query = self.pos_enc(query)

        # Causal mask for self-attention in decoder
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=z.device,
                                     dtype=torch.bool), diagonal=1)
        out = self.decoder(tgt=query, memory=memory, tgt_mask=mask)    # (N, T, d_model)
        out = self.output_map(out)                                     # (N, T, 7)
        X_hat = torch.clamp(out.permute(0, 2, 1), 0, 1)
        return X_hat

    def _predict_max_vals(self, z):
        N = z.size(0)
        mvtp = self.max_value_predictor(z)
        mvp = self.inverse_transform_max_values(mvtp)
        ones = torch.ones(N, 1, device=z.device)
        max_vals_pred = torch.cat([ones, mvp], dim=1)
        ones_t = self.transform_max_values(ones)
        max_vals_pred_t = torch.cat([ones_t, mvtp], dim=1)
        return max_vals_pred, max_vals_pred_t

    def forward(self, X, max_vals=None, teacher_forcing_ratio=0.5):
        mu, log_var = self._encode(X, max_vals)
        z = self.sample(mu, log_var)
        max_vals_pred, max_vals_pred_t = self._predict_max_vals(z)
        X_hat = self._decode_from_z(z)
        return X_hat, mu, log_var, max_vals_pred, max_vals_pred_t

    def generate(self, num_samples, device):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            max_vals_pred, _ = self._predict_max_vals(z)
            X_hat = self._decode_from_z(z)
            return X_hat, max_vals_pred

    def decode(self, z):
        self.eval()
        with torch.no_grad():
            max_vals_pred, _ = self._predict_max_vals(z)
            X_hat = self._decode_from_z(z)
            return X_hat, max_vals_pred
