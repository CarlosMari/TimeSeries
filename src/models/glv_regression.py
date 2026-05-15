"""Direct GLV regression model (model 7 in the pivot lineup).

Architecture: BiLSTM encoder → MLP → (r̂ ∈ ℝ^7, Â ∈ ℝ^{7×7}). At inference
("generation") time, we integrate the predicted system with scipy.solve_ivp,
yielding a trajectory.

Key differences from the VAE-family models 1–6:

  - No latent space, no KL divergence, no variational objective.
  - Trains on the **matched dataset** (PARAM_RECOVERY_MATCHED.pkl, 10k samples)
    where ground-truth (r, A) is available.
  - "Reconstruction" = integrate (r̂, Â) → trajectory; compare to ground truth.
  - "Generation" = sample (r, A) from the train-time empirical distribution,
    integrate, return.

Conceptual role: this is the **physics-naive inverse-problem baseline**.
A reviewer asking "if you can recover the parameters, isn't that just an
inverse problem?" is answered by including model 7 as a direct comparator.
By construction it should win on parameter recovery and lose on flexibility.

API: same forward / generate / decode signature as LSTM_VAE so the eval
harness can call it the same way. For generate(), the latent_dim placeholder
is the dimensionality of the sampling distribution we use for (r, A) — but
since there's no real latent space, the "z" input is interpreted as a
seed for the empirical sampler.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

ACTIVATION = nn.SiLU()


class GLVRegressor(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        self.n_curves = config.get("n_curves", 7)
        self.seq_len = config.get("seq_len", 65)
        # Latent dim is unused in this model but we keep the attribute for
        # API compatibility (the eval harness inspects it).
        self.latent_dim = config.get("latent_dim", 30)

        self.rnn_hidden_size = config.get("rnn_hidden_size", 256)
        self.rnn_num_layers = config.get("rnn_num_layers", 2)
        self.t_final = config.get("t_final", 12.8)   # matches v1 simulation horizon
        self.solver = config.get("solver", "RK45")

        print(f"GLVRegressor: hidden={self.rnn_hidden_size} layers={self.rnn_num_layers}")
        print(f"  Integration: t in [0, {self.t_final}], solver={self.solver}, T={self.seq_len}")

        self.encoder_rnn = nn.LSTM(
            input_size=self.n_curves,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        encoder_output_dim = self.rnn_num_layers * 2 * self.rnn_hidden_size

        self.head = nn.Sequential(
            nn.Linear(encoder_output_dim, 512),
            ACTIVATION,
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            ACTIVATION,
            nn.Linear(256, self.n_curves + self.n_curves * self.n_curves),
        )

        # We'll cache an empirical dataset of (r, A, x0) for unconditional
        # generation. Populated by ``store_empirical_distribution`` after
        # training; the unified eval harness calls it.
        self._emp_r = None
        self._emp_A = None
        self._emp_x0 = None

    # ------------------------------------------------------------------
    # Parameter regression
    # ------------------------------------------------------------------
    def regress(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """X: (N, 7, T) → (r̂: (N, 7), Â: (N, 7, 7))."""
        N = X.size(0)
        X_rnn = X.permute(0, 2, 1)
        _, (h_n, _) = self.encoder_rnn(X_rnn)
        encoded = h_n.permute(1, 0, 2).contiguous().view(N, -1)
        out = self.head(encoded)
        r_hat = out[:, : self.n_curves]
        A_hat = out[:, self.n_curves:].view(N, self.n_curves, self.n_curves)
        return r_hat, A_hat

    # ------------------------------------------------------------------
    # Integration (numpy, sequential; CPU)
    # ------------------------------------------------------------------
    def integrate_glv(self, r: np.ndarray, A: np.ndarray, x0: np.ndarray,
                      t_final: float = None, T: int = None) -> np.ndarray:
        """Integrate dx_i/dt = x_i (r_i + sum_j A_ij x_j) on [0, t_final].

        r: (7,), A: (7, 7), x0: (7,). Returns (7, T)."""
        if t_final is None:
            t_final = self.t_final
        if T is None:
            T = self.seq_len
        t_eval = np.linspace(0.0, t_final, T)

        def rhs(t, x):
            return x * (r + A @ x)

        sol = solve_ivp(rhs, [0.0, t_final], x0, t_eval=t_eval, method=self.solver,
                        rtol=1e-5, atol=1e-7)
        if not sol.success:
            return np.full((self.n_curves, T), np.nan, dtype=np.float64)
        y = sol.y  # (7, T)
        return np.maximum(y, 0.0)

    # ------------------------------------------------------------------
    # API compat with VAE-family models
    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, max_vals: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5):
        """Return ((r̂, Â) reshaped into an X-like tensor for compat, plus dummies).

        We can't reconstruct trajectories on-GPU with solve_ivp, so during
        training we return only the regressed parameters as an Nx*-tensor and
        the loss function is overridden externally. To stay compatible with
        train_cvae.train(), we return zeros for the "X_hat" slot.
        """
        r_hat, A_hat = self.regress(X)
        # API-compatible padding: zero "X_hat", zero "mu/log_var/max_vals_pred"
        N = X.size(0)
        zero_X = torch.zeros_like(X)
        zero_mu = torch.zeros(N, self.latent_dim, device=X.device)
        zero_lv = torch.zeros(N, self.latent_dim, device=X.device)
        # Use the predicted max(A_hat) as a stand-in for max_vals_pred (just so
        # the unified eval harness has a "max_vals" attribute to read; it won't
        # be used since this model doesn't have a meaningful max-value head).
        ones = torch.ones(N, self.n_curves, device=X.device)
        return zero_X, zero_mu, zero_lv, ones, ones

    def generate(self, num_samples: int, device: torch.device):
        """Sample (r, A, x0) from the empirical distribution and integrate.

        Returns (X: (N, 7, T), max_vals: (N, 7)). max_vals is *the true peak
        per curve* after the integration, since this model has no scale
        prediction head — the scale is whatever solve_ivp produces.
        """
        self.eval()
        if self._emp_r is None:
            raise RuntimeError(
                "GLVRegressor.generate() requires an empirical distribution; "
                "call store_empirical_distribution() with training-set (r, A, x0) first."
            )
        rng = np.random.default_rng()
        idx = rng.integers(0, self._emp_r.shape[0], size=num_samples)
        Xs = []
        max_vals = []
        for i in idx:
            x_traj = self.integrate_glv(self._emp_r[i], self._emp_A[i], self._emp_x0[i])
            Xs.append(x_traj)
            max_vals.append(x_traj.max(axis=1))
        X = np.stack(Xs, axis=0)
        mv = np.stack(max_vals, axis=0)
        return (torch.from_numpy(X).float().to(device),
                torch.from_numpy(mv).float().to(device))

    def decode(self, z: torch.Tensor):
        """For API compat. z is used as a seed for empirical sampling; the
        "latent" doesn't correspond to a structured representation here.
        """
        self.eval()
        return self.generate(z.size(0), z.device)

    # ------------------------------------------------------------------
    # Empirical-distribution caching for generation
    # ------------------------------------------------------------------
    def store_empirical_distribution(self, r: np.ndarray, A: np.ndarray, x0: np.ndarray):
        """Register the training-set empirical distribution of (r, A, x0)."""
        assert r.shape[1] == self.n_curves
        assert A.shape[1] == A.shape[2] == self.n_curves
        assert x0.shape[1] == self.n_curves
        self._emp_r = r.astype(np.float32)
        self._emp_A = A.astype(np.float32)
        self._emp_x0 = x0.astype(np.float32)
