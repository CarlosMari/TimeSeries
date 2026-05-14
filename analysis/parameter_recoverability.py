"""
Parameter recoverability experiment (PLAN.md Option B).

Question: Does the latent space mu(z) implicitly identify the GLV parameters
(r, A) of the underlying dynamical system?

Procedure:
1. Generate a fresh matched dataset of N samples where we record both the
   trajectory and the (r, A) used to produce it. Use seeds well outside the
   train/test ranges so this is held-out from the trained model.
2. Apply the same 3-stage preprocessing pipeline used at training time
   (family-norm, sort by peak, per-curve norm). Track the sort permutation
   per sample so we can permute (r, A) consistently — otherwise the
   regression target is scrambled relative to the model's species ordering.
3. Encode trajectories through the trained CVAE to get mu (N, 30).
4. Fit Ridge regression mu -> r (7 outputs) and mu -> vec(A) (49 outputs)
   on a train split, evaluate held-out R^2 per parameter.

Outcomes are publishable either way:
  - Strong (mean R^2 >= 0.5): "latent space implicitly identifies GLV
    parameters — implicit system identification from trajectories alone."
  - Moderate (0.2 <= R^2 < 0.5): "latent space encodes partial information
    about underlying parameters."
  - Weak (R^2 < 0.2): "latent space encodes dynamical phenotypes, not
    parameters" — the negative result is itself informative and dovetails
    with the species-centric interpretability story already in the paper.

Writes:
  data/PARAM_RECOVERY_MATCHED.pkl  -- the matched dataset (trajectories +
                                       params + processed copies)
  RESULTS_PARAM_RECOVERY.json      -- recoverability metrics
  final figures/fig_param_recoverability.{pdf,png}

The generation step is deterministic from the SEED constant; re-running this
script reproduces the same dataset.
"""

from __future__ import annotations

import json
import pickle
import signal
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_generation"))

from scipy import integrate  # noqa: E402

from src.models.cvae import LSTM_VAE  # noqa: E402

# Configuration
MATCHED_SEED_BASE = 555_000_001  # outside TRAIN (123456789 + 1M+i) and TEST (987654321 + 1M+i) ranges
N_TARGET = 10_000               # 10k samples is plenty for Ridge on 30D inputs / 49D outputs
TIMEOUT_PER_SEED = 20           # seconds
SIGMA = 0.01                    # matches the noise level in the original pipeline
N_SPECIES = 7
SEQ_LEN = 65

OUT_DATA = REPO_ROOT / "data" / "PARAM_RECOVERY_MATCHED.pkl"
OUT_JSON = REPO_ROOT / "RESULTS_PARAM_RECOVERY.json"
OUT_FIG = REPO_ROOT / "final figures" / "fig_param_recoverability.pdf"

MODEL_PATH = REPO_ROOT / "model_ckpts" / "model_final_30_conditioned.pth"
MODEL_CONFIG = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": "log",
    "use_scale_conditioning": True,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 1. Generation with parameter recording
# ----------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _lotka_volterra_rhs(t, x, r, A_T):
    """dx/dt = r * x + x * (A_T @ x). A_T is A transposed to match the
    convention in custom_glv_FIXED (params reshape A column-major)."""
    return r * x + x * (A_T @ x)


def _generate_one(seed: int):
    """Reproduces custom_glv_FIXED.generate_curves_Mario logic but ALSO
    returns (r, A). Single RNG controls both parameter sampling and the
    initial condition. Noise is added with a deterministic offset RNG so the
    full output is reproducible from `seed`. Returns (traj, r, A) or None.
    """
    rng = np.random.RandomState(seed)

    flag = False
    c = 0
    max_attempts = 500
    r0 = a0 = xss = None

    while not flag and c < max_attempts:
        r0 = rng.exponential(scale=2.0, size=N_SPECIES)
        a0 = rng.randn(N_SPECIES, N_SPECIES)
        for i in range(N_SPECIES):
            a0[i, i] = -rng.exponential(scale=2.0)

        try:
            xss = np.linalg.solve(a0, -r0)
            d0 = np.diag(xss)
            eig = np.real(np.linalg.eigvals(d0 @ a0))
            flag = np.all(eig <= 0) and np.all(xss > 0)
        except np.linalg.LinAlgError:
            flag = False
        c += 1

    if not flag:
        return None

    # Same initial condition draw as custom_glv_FIXED
    initial_condition = rng.exponential(scale=0.1, size=N_SPECIES)

    # Same time grid as the training pipeline (generate_family_FIXED.py passes tmax=20)
    tmax = 20.0
    times = np.linspace(0.0, tmax, SEQ_LEN)

    # Match custom_glv_FIXED: params = [r0, a0.T.flatten()], then
    # b = params[K:].reshape((K, K)).T  =>  b = a0
    A_T = a0  # reshape-then-transpose yields the original a0 matrix
    try:
        sol = integrate.solve_ivp(
            fun=lambda t, y: _lotka_volterra_rhs(t, y, r0, A_T),
            t_span=(0.0, tmax),
            y0=initial_condition,
            t_eval=times,
            method="RK45",
            max_step=0.5,
        )
        if not sol.success:
            return None
        traj = sol.y  # (7, 65)
    except Exception:
        return None

    # Apply the same lognormal noise as generate_data (deterministic from seed)
    noise_rng = np.random.RandomState(seed + 7)
    scaling_factor = float(np.exp(SIGMA ** 2 / 2))
    noise = noise_rng.lognormal(mean=0, sigma=SIGMA, size=traj.shape)
    traj = traj * (noise / scaling_factor)

    # Match the quality checks used in generate_family_FIXED.py
    steady_states = np.mean(traj[:, -10:], axis=1)
    overshoot = np.sum(np.max(traj, axis=1) > 1.2 * steady_states)
    if (
        np.isnan(traj).any()
        or (traj > 3.0).any()
        or (np.max(traj, axis=1) < 0.1).any()
        or overshoot < 3
    ):
        return None

    return traj, r0, a0


def generate_matched_dataset(n_target: int, seed_base: int):
    """Generate n_target accepted (trajectory, r, A) triples."""
    if OUT_DATA.exists():
        print(f"Found existing dataset at {OUT_DATA}, loading...")
        with open(OUT_DATA, "rb") as f:
            return pickle.load(f)

    trajs, rs, As = [], [], []
    n_attempted = 0
    n_accepted = 0
    n_timeouts = 0
    t0 = time.time()

    signal.signal(signal.SIGALRM, _alarm_handler)

    while n_accepted < n_target:
        signal.alarm(TIMEOUT_PER_SEED)
        try:
            out = _generate_one(seed_base + n_attempted)
        except _Timeout:
            out = None
            n_timeouts += 1
        finally:
            signal.alarm(0)
        n_attempted += 1
        if out is not None:
            traj, r0, a0 = out
            trajs.append(traj)
            rs.append(r0)
            As.append(a0)
            n_accepted += 1
            if n_accepted % 500 == 0:
                elapsed = time.time() - t0
                rate = n_accepted / max(elapsed, 1e-9)
                eta = (n_target - n_accepted) / max(rate, 1e-9)
                print(
                    f"  {n_accepted}/{n_target} accepted "
                    f"(attempted {n_attempted}, timeouts {n_timeouts}, "
                    f"{rate:.1f}/s, ETA {eta/60:.1f}min)"
                )

    trajs = np.stack(trajs, axis=0)  # (N, 7, 65)
    rs = np.stack(rs, axis=0)        # (N, 7)
    As = np.stack(As, axis=0)        # (N, 7, 7)

    # ------------------------------------------------------------------------
    # Apply the same preprocessing pipeline used at training time:
    # 1. family_max normalization
    # 2. sort curves by peak (descending) — RECORD THE PERMUTATION
    # 3. per-curve max normalization
    # ------------------------------------------------------------------------
    family_max = np.max(trajs, axis=(1, 2), keepdims=True)  # (N,1,1)
    family_max = np.where(family_max == 0, 1e-8, family_max)
    fam_normalized = trajs / family_max

    max_for_sort = np.max(fam_normalized, axis=2)  # (N, 7)
    perm = np.argsort(-max_for_sort, axis=1)        # (N, 7) - sorted indices
    sorted_data = np.take_along_axis(
        fam_normalized, perm[:, :, np.newaxis], axis=1
    )

    per_curve_max = np.max(sorted_data, axis=2, keepdims=True)
    per_curve_max = np.where(per_curve_max == 0, 1e-8, per_curve_max)
    final_data = sorted_data / per_curve_max

    pkg = {
        "data": final_data,                                          # (N,7,65)  model input
        "reconstruction_max_values": np.squeeze(per_curve_max, axis=2),  # (N,7)
        "family_max_values": np.squeeze(family_max, axis=(1, 2)),    # (N,)
        "raw_trajectories": trajs,                                   # (N,7,65)
        "r": rs,                                                     # (N,7)  unsorted!
        "A": As,                                                     # (N,7,7) unsorted!
        "sort_permutation": perm,                                    # (N,7)
        "seed_base": seed_base,
        "n_attempted": n_attempted,
        "n_timeouts": n_timeouts,
    }
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DATA, "wb") as f:
        pickle.dump(pkg, f)
    print(f"Saved matched dataset → {OUT_DATA}")
    return pkg


# ----------------------------------------------------------------------------
# 2. Encode through the trained CVAE → mu(z)
# ----------------------------------------------------------------------------


def encode_mus(pkg):
    print("Loading CVAE...")
    model = LSTM_VAE(MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    X = pkg["data"]
    mv = pkg["reconstruction_max_values"]
    N = X.shape[0]

    mus = np.empty((N, MODEL_CONFIG["latent_dim"]), dtype=np.float32)
    bs = 1000
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=DEVICE)
            mb = torch.tensor(mv[i:i+bs], dtype=torch.float32, device=DEVICE)
            _, mu, _, _, _ = model(xb, mb, teacher_forcing_ratio=0.0)
            mus[i:i+bs] = mu.cpu().numpy()
    return mus


# ----------------------------------------------------------------------------
# 3. Build aligned targets (apply sort permutation to r and A)
# ----------------------------------------------------------------------------


def align_params_with_sort(pkg):
    """The trained model sees curves sorted by peak. The regression target
    must respect that order, so we permute r and A consistently.

    r_sorted[i] = r[perm[i]]              (length-7 vector permuted)
    A_sorted[i,j] = A[perm[i], perm[j]]   (matrix row+col permuted)
    """
    r = pkg["r"]                # (N, 7)
    A = pkg["A"]                # (N, 7, 7)
    perm = pkg["sort_permutation"]  # (N, 7)
    N = r.shape[0]

    r_sorted = np.take_along_axis(r, perm, axis=1)
    A_sorted = np.empty_like(A)
    for n in range(N):
        p = perm[n]
        A_sorted[n] = A[n][np.ix_(p, p)]

    return r_sorted, A_sorted


# ----------------------------------------------------------------------------
# 4. Ridge regression with K-fold CV on held-out predictions
# ----------------------------------------------------------------------------


def kfold_predict(X, Y, n_splits=5, alpha=1.0):
    """Returns out-of-fold predictions Y_hat of shape == Y.shape."""
    Y_hat = np.empty_like(Y, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr, te in kf.split(X):
        reg = Ridge(alpha=alpha)
        reg.fit(X[tr], Y[tr])
        Y_hat[te] = reg.predict(X[te])
    return Y_hat


def evaluate(mus, r_sorted, A_sorted):
    print("Fitting Ridge for r (7 outputs)...")
    r_hat = kfold_predict(mus, r_sorted)
    r2_r_per = [r2_score(r_sorted[:, k], r_hat[:, k]) for k in range(7)]
    r2_r_overall = float(np.mean(r2_r_per))

    print("Fitting Ridge for vec(A) (49 outputs)...")
    A_flat = A_sorted.reshape(A_sorted.shape[0], -1)
    A_hat_flat = kfold_predict(mus, A_flat)
    r2_A_per = [r2_score(A_flat[:, k], A_hat_flat[:, k]) for k in range(49)]
    r2_A_overall = float(np.mean(r2_A_per))

    # Diagonal of A (self-interaction / carrying-capacity-like): typically the
    # easiest to recover, since it sets the trajectory shape most strongly.
    diag_idx = [i * 7 + i for i in range(7)]
    r2_A_diag = float(np.mean([r2_A_per[i] for i in diag_idx]))
    offdiag_idx = [i for i in range(49) if i not in diag_idx]
    r2_A_offdiag = float(np.mean([r2_A_per[i] for i in offdiag_idx]))

    # Eigenvalues of A as a derived target — capture stability/oscillation info
    print("Fitting Ridge for eigenvalues of diag(x*) A (system spectrum)...")
    # Compute target: real & imag parts of eigenvalues of A
    N = A_sorted.shape[0]
    eigs = np.empty((N, 7), dtype=np.complex128)
    for n in range(N):
        eigs[n] = np.linalg.eigvals(A_sorted[n])
    # Sort eigenvalues by real part (descending) for consistent target
    idx = np.argsort(-eigs.real, axis=1)
    eigs = np.take_along_axis(eigs, idx, axis=1)
    target_real = eigs.real
    target_imag = eigs.imag

    real_hat = kfold_predict(mus, target_real)
    imag_hat = kfold_predict(mus, target_imag)
    r2_eig_real = float(np.mean([r2_score(target_real[:, k], real_hat[:, k]) for k in range(7)]))
    r2_eig_imag = float(np.mean([r2_score(target_imag[:, k], imag_hat[:, k]) for k in range(7)]))

    return {
        "r_R2_per_species": r2_r_per,
        "r_R2_mean": r2_r_overall,
        "A_R2_per_entry": r2_A_per,
        "A_R2_mean": r2_A_overall,
        "A_R2_diag_mean": r2_A_diag,
        "A_R2_offdiag_mean": r2_A_offdiag,
        "eig_real_R2_mean": r2_eig_real,
        "eig_imag_R2_mean": r2_eig_imag,
    }


# ----------------------------------------------------------------------------
# 5. Figure
# ----------------------------------------------------------------------------


def make_figure(results, r_sorted, r_hat, A_sorted, A_hat_flat):
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # A) Bar plot of R^2 per parameter group
    ax = axes[0, 0]
    bars = [
        ("Growth rate r", results["r_R2_mean"]),
        ("A: diagonal", results["A_R2_diag_mean"]),
        ("A: off-diag", results["A_R2_offdiag_mean"]),
        ("A: all", results["A_R2_mean"]),
        ("Re(eig(A))", results["eig_real_R2_mean"]),
        ("Im(eig(A))", results["eig_imag_R2_mean"]),
    ]
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = ["tab:blue" if v >= 0.5 else "tab:orange" if v >= 0.2 else "tab:red" for v in vals]
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(0.2, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.axvline(0.5, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlabel(r"$R^2$ (out-of-fold)")
    ax.set_title("A) Parameter recoverability from $\\mu(z)$")
    ax.set_xlim(min(0, min(vals) - 0.05), 1)

    # B) r scatter (one example species)
    ax = axes[0, 1]
    k = int(np.argmax(results["r_R2_per_species"]))
    ax.scatter(r_sorted[:, k], r_hat[:, k], s=4, alpha=0.3)
    lo = min(r_sorted[:, k].min(), r_hat[:, k].min())
    hi = max(r_sorted[:, k].max(), r_hat[:, k].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"true $r_{k}$")
    ax.set_ylabel(f"predicted $r_{k}$")
    ax.set_title(f"B) Best species growth rate — $R^2$={results['r_R2_per_species'][k]:.3f}")

    # C) A entry scatter (most recoverable entry)
    ax = axes[0, 2]
    k = int(np.argmax(results["A_R2_per_entry"]))
    i, j = divmod(k, 7)
    ax.scatter(A_sorted.reshape(-1, 49)[:, k], A_hat_flat[:, k], s=4, alpha=0.3)
    lo = min(A_sorted.reshape(-1, 49)[:, k].min(), A_hat_flat[:, k].min())
    hi = max(A_sorted.reshape(-1, 49)[:, k].max(), A_hat_flat[:, k].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"true $A_{{{i}{j}}}$")
    ax.set_ylabel(f"predicted $A_{{{i}{j}}}$")
    ax.set_title(f"C) Best $A$ entry — $R^2$={results['A_R2_per_entry'][k]:.3f}")

    # D) Heatmap of per-entry A R^2
    ax = axes[1, 0]
    A_r2_matrix = np.array(results["A_R2_per_entry"]).reshape(7, 7)
    im = ax.imshow(A_r2_matrix, cmap="RdYlGn", vmin=-0.5, vmax=1)
    plt.colorbar(im, ax=ax, label="$R^2$")
    ax.set_title("D) Per-entry $R^2$ for $A_{ij}$")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    for ii in range(7):
        for jj in range(7):
            ax.text(jj, ii, f"{A_r2_matrix[ii, jj]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(A_r2_matrix[ii, jj]) < 0.5 else "white")

    # E) Per-species r R^2 bar
    ax = axes[1, 1]
    ax.bar(range(7), results["r_R2_per_species"])
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlabel("species (sorted by peak)")
    ax.set_ylabel("$R^2$")
    ax.set_title("E) Per-species growth-rate $R^2$")

    # F) Summary verdict text
    ax = axes[1, 2]
    ax.axis("off")
    verdict_lines = [
        "Verdict (out-of-fold $R^2$, 5-fold CV):",
        "",
        f"  r̄: {results['r_R2_mean']:.3f}",
        f"  A diag: {results['A_R2_diag_mean']:.3f}",
        f"  A off-diag: {results['A_R2_offdiag_mean']:.3f}",
        f"  Re(eig A): {results['eig_real_R2_mean']:.3f}",
        f"  Im(eig A): {results['eig_imag_R2_mean']:.3f}",
        "",
        ("Strong recoverability (>0.5)" if results["r_R2_mean"] >= 0.5
         else ("Partial recoverability (>0.2)" if results["r_R2_mean"] >= 0.2
               else "Weak recoverability (<0.2)")),
        "",
        "Interpretation in PROJECT.md §4.",
    ]
    ax.text(0.02, 0.95, "\n".join(verdict_lines), va="top",
            family="monospace", fontsize=9)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved figure → {OUT_FIG}")


def main():
    print("Step 1/4: generating matched dataset (or loading if cached)...")
    pkg = generate_matched_dataset(N_TARGET, MATCHED_SEED_BASE)
    print(f"  Dataset shape: {pkg['data'].shape}")

    print("Step 2/4: encoding through CVAE...")
    mus = encode_mus(pkg)

    print("Step 3/4: aligning targets with sort permutation...")
    r_sorted, A_sorted = align_params_with_sort(pkg)

    print("Step 4/4: evaluating Ridge regression with K-fold CV...")
    results = evaluate(mus, r_sorted, A_sorted)

    # Compute predictions one more time for the plot
    r_hat = kfold_predict(mus, r_sorted)
    A_hat_flat = kfold_predict(mus, A_sorted.reshape(A_sorted.shape[0], -1))

    print("\n=== RESULTS ===")
    print(f"r̄ mean R²       : {results['r_R2_mean']:.4f}")
    print(f"r per species   : {[round(v,3) for v in results['r_R2_per_species']]}")
    print(f"A all mean R²   : {results['A_R2_mean']:.4f}")
    print(f"A diag mean R²  : {results['A_R2_diag_mean']:.4f}")
    print(f"A off-diag R²   : {results['A_R2_offdiag_mean']:.4f}")
    print(f"Re(eig A) R²    : {results['eig_real_R2_mean']:.4f}")
    print(f"Im(eig A) R²    : {results['eig_imag_R2_mean']:.4f}")

    OUT_JSON.write_text(json.dumps({
        "n_samples": int(mus.shape[0]),
        "latent_dim": mus.shape[1],
        **results,
    }, indent=2))
    print(f"Saved → {OUT_JSON}")

    make_figure(results, r_sorted, r_hat, A_sorted, A_hat_flat)


if __name__ == "__main__":
    main()
