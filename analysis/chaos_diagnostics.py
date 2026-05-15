"""
Chaos diagnostics for the paper (Phase 3 in PLAN.md).

Goal: provide CSF-reviewer-expected nonlinear-dynamics analyses beyond the
recurrence plots we already have. We compute two standard NLD diagnostics on
matched real vs generated trajectories and compare distributions with a
two-sample KS test:

  1. **Recurrence Quantification Analysis (RQA)** — recurrence rate (RR),
     determinism (DET), mean diagonal line length (L_mean), maximum diagonal
     line length (L_max), laminarity (LAM), trapping time (TT).

  2. **Largest Lyapunov exponent** via Rosenstein's algorithm.

Sequences are short (T=65). We declare this as a limitation; the goal is
*distributional* comparison real vs generated, not a precision measurement
of any single trajectory.

Inputs
------
- model_ckpts/model_final_30_conditioned.pth  (30D scale-conditioned CVAE)
- data/TEST_FINAL_PROCESSED.pkl                (real held-out test set)

Outputs
-------
- RESULTS_CHAOS.json                            (numbers + per-measure KS)
- final figures/fig_chaos_diagnostics.{pdf,png} (single multi-panel figure)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.cvae import LSTM_VAE  # noqa: E402
from utils.post_processing import apply_extinction_threshold  # noqa: E402

MODEL_PATH = REPO_ROOT / "model_ckpts" / "model_final_30_conditioned.pth"
TEST_PATH = REPO_ROOT / "data" / "TEST_FINAL_PROCESSED.pkl"

OUT_JSON = REPO_ROOT / "RESULTS_CHAOS.json"
OUT_FIG = REPO_ROOT / "final figures" / "fig_chaos_diagnostics.pdf"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 200            # per group (real and generated) — RQA is O(T^2) per traj
RNG = np.random.default_rng(2026_05_15)

MODEL_CONFIG = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": "log",
    "use_scale_conditioning": True,
}

# RQA parameters. With T=65 we use a small embedding so the recurrence matrix
# stays usable (N_rp = T - (m-1)*tau).
RQA_EMBED_DIM = 3
RQA_EMBED_TAU = 2
RQA_RECURRENCE_RATE_TARGET = 0.10   # pick eps so RR ≈ 10% per trajectory
RQA_MIN_LINE_LENGTH = 2

# Rosenstein parameters
LYAP_EMBED_DIM = 3
LYAP_EMBED_TAU = 2
LYAP_FIT_FRACTION = 0.5             # fit slope on first half of divergence curve
LYAP_MEAN_PERIOD = 5                # Theiler window: skip neighbors within this many steps


# ---------------------------------------------------------------------------
# Helpers: time-delay embedding
# ---------------------------------------------------------------------------

def time_delay_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """1-D signal x → embedded trajectory of shape (N - (m-1)*tau, m)."""
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    out = np.empty((n, m))
    for i in range(m):
        out[:, i] = x[i * tau : i * tau + n]
    return out


def species_avg_signal(traj: np.ndarray) -> np.ndarray:
    """Collapse a (7, T) trajectory to a 1-D signal by averaging across species.

    We use the mean over species (each species is normalized to peak=1 within
    family, then sorted, so the mean captures the system-level dynamics
    without any one species dominating). This is the same idea recurrence
    plots use in the paper's existing fig_recurrence_dynamics.
    """
    return traj.mean(axis=0)


# ---------------------------------------------------------------------------
# 1. Recurrence Quantification Analysis
# ---------------------------------------------------------------------------

def _recurrence_matrix(embedded: np.ndarray, eps: float) -> np.ndarray:
    D = squareform(pdist(embedded, metric="euclidean"))
    return (D <= eps).astype(np.uint8)


def _choose_eps_for_rr(embedded: np.ndarray, target_rr: float) -> float:
    """Pick eps so the recurrence rate is approximately `target_rr`."""
    D = pdist(embedded, metric="euclidean")
    if D.size == 0:
        return 0.0
    return float(np.quantile(D, target_rr))


def _line_lengths(mat: np.ndarray, diagonal: bool) -> list[int]:
    """Return the list of run-lengths of 1s along diagonals (LOI excluded) or
    along vertical lines.
    """
    N = mat.shape[0]
    runs: list[int] = []
    if diagonal:
        # all diagonals except the main one
        for k in range(1, N):
            d = np.diag(mat, k=k)
            run = 0
            for v in d:
                if v:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                    run = 0
            if run > 0:
                runs.append(run)
            # symmetric matrix → also count the −k diagonal
            d = np.diag(mat, k=-k)
            run = 0
            for v in d:
                if v:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                    run = 0
            if run > 0:
                runs.append(run)
    else:
        for j in range(N):
            col = mat[:, j]
            run = 0
            for v in col:
                if v:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                    run = 0
            if run > 0:
                runs.append(run)
    return runs


def rqa_measures(x: np.ndarray) -> dict:
    """Compute RQA measures for a 1-D signal x.

    Returns dict with keys RR, DET, L_mean, L_max, LAM, TT.
    """
    emb = time_delay_embed(x, m=RQA_EMBED_DIM, tau=RQA_EMBED_TAU)
    if emb.shape[0] < 4:
        return {k: float("nan") for k in ("RR", "DET", "L_mean", "L_max", "LAM", "TT")}

    eps = _choose_eps_for_rr(emb, RQA_RECURRENCE_RATE_TARGET)
    R = _recurrence_matrix(emb, eps)

    N = R.shape[0]
    # Recurrence rate (exclude line of identity for honesty)
    R_loi = R.copy()
    np.fill_diagonal(R_loi, 0)
    rr = float(R_loi.sum()) / (N * (N - 1))

    diag_runs = _line_lengths(R_loi, diagonal=True)
    vert_runs = _line_lengths(R_loi, diagonal=False)

    diag_runs_arr = np.array([r for r in diag_runs if r >= RQA_MIN_LINE_LENGTH], dtype=np.int64)
    vert_runs_arr = np.array([r for r in vert_runs if r >= RQA_MIN_LINE_LENGTH], dtype=np.int64)

    total_recurrent_diag = sum(diag_runs)  # all run points (length 1 included)
    det = float(diag_runs_arr.sum()) / total_recurrent_diag if total_recurrent_diag else 0.0

    l_mean = float(diag_runs_arr.mean()) if diag_runs_arr.size else 0.0
    l_max = int(diag_runs_arr.max()) if diag_runs_arr.size else 0

    total_recurrent_vert = sum(vert_runs)
    lam = float(vert_runs_arr.sum()) / total_recurrent_vert if total_recurrent_vert else 0.0
    tt = float(vert_runs_arr.mean()) if vert_runs_arr.size else 0.0

    return {
        "RR": rr,
        "DET": det,
        "L_mean": l_mean,
        "L_max": l_max,
        "LAM": lam,
        "TT": tt,
    }


# ---------------------------------------------------------------------------
# 2. Largest Lyapunov exponent (Rosenstein 1993)
# ---------------------------------------------------------------------------

def rosenstein_lyapunov(x: np.ndarray,
                        m: int = LYAP_EMBED_DIM,
                        tau: int = LYAP_EMBED_TAU,
                        theiler: int = LYAP_MEAN_PERIOD,
                        fit_frac: float = LYAP_FIT_FRACTION) -> float:
    """Estimate the largest Lyapunov exponent via Rosenstein's method.

    Steps:
      1. Embed the signal in m dimensions with delay tau.
      2. For each point, find nearest neighbor with |index difference| > theiler.
      3. Track the average log-distance d(k) between each pair after k steps.
      4. Fit a straight line to the early part of <ln d(k)> vs k; slope = λ_1.

    Returns slope per timestep. For very short signals this is noisy; we
    return it anyway and rely on distributional comparison.
    """
    emb = time_delay_embed(x, m=m, tau=tau)
    N = emb.shape[0]
    if N < 8:
        return float("nan")

    D = squareform(pdist(emb, metric="euclidean"))
    np.fill_diagonal(D, np.inf)

    # Apply Theiler window: forbid temporally close neighbors
    idx = np.arange(N)
    forbid = np.abs(idx[:, None] - idx[None, :]) <= theiler
    D_masked = D.copy()
    D_masked[forbid] = np.inf
    nn = np.argmin(D_masked, axis=1)

    # If any neighbor is still inf (no valid neighbor), drop those points
    valid = D_masked[idx, nn] < np.inf
    if valid.sum() < 4:
        return float("nan")

    max_steps = N - 1 - max(idx[valid].max(), nn[valid].max())
    if max_steps < 4:
        max_steps = max(4, min(N - 1, 12))

    # Average log-distance at each follow-up step k
    log_d = []
    for k in range(max_steps):
        i = idx[valid]
        j = nn[valid]
        ip = i + k
        jp = j + k
        m_ok = (ip < N) & (jp < N)
        if m_ok.sum() < 4:
            break
        d = np.linalg.norm(emb[ip[m_ok]] - emb[jp[m_ok]], axis=1)
        d = d[d > 0]
        if d.size < 4:
            break
        log_d.append(np.mean(np.log(d)))
    log_d = np.array(log_d)
    if log_d.size < 4:
        return float("nan")

    # Fit slope on the first fit_frac fraction of the divergence curve.
    n_fit = max(3, int(len(log_d) * fit_frac))
    ks = np.arange(n_fit)
    slope, _ = np.polyfit(ks, log_d[:n_fit], 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def compute_for_population(trajs: np.ndarray, label: str) -> dict:
    """trajs (N, 7, T). Returns dict with stacked per-traj RQA + Lyapunov."""
    rqa_keys = ("RR", "DET", "L_mean", "L_max", "LAM", "TT")
    rqa = {k: [] for k in rqa_keys}
    lyap = []
    for i in range(trajs.shape[0]):
        sig = species_avg_signal(trajs[i])
        m = rqa_measures(sig)
        for k in rqa_keys:
            rqa[k].append(m[k])
        lyap.append(rosenstein_lyapunov(sig))
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i + 1}/{trajs.shape[0]}")
    return {
        "rqa": {k: np.array(v, dtype=float) for k, v in rqa.items()},
        "lyap": np.array(lyap, dtype=float),
    }


def summarize(arr: np.ndarray) -> dict:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "n": 0}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def ks_compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return {"ks_stat": float("nan"), "ks_p": float("nan"),
                "n_real": int(a.size), "n_gen": int(b.size)}
    stat, p = stats.ks_2samp(a, b)
    return {"ks_stat": float(stat), "ks_p": float(p),
            "n_real": int(a.size), "n_gen": int(b.size)}


def main():
    print("Loading model & test data...")
    model = LSTM_VAE(MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with open(TEST_PATH, "rb") as f:
        d = pickle.load(f)
    real_norm = d["data"]
    real_mv = d["reconstruction_max_values"]
    real_fam = d["family_max_values"]

    real_idx = RNG.choice(real_norm.shape[0], size=N_SAMPLES, replace=False)
    real_orig = (
        real_norm[real_idx]
        * real_mv[real_idx][:, :, None]
        * real_fam[real_idx][:, None, None]
    )

    print(f"Generating {N_SAMPLES} samples from the trained model...")
    torch.manual_seed(20260515)
    with torch.no_grad():
        z = torch.randn(N_SAMPLES, MODEL_CONFIG["latent_dim"], device=DEVICE)
        gen_norm, gen_mv = model.decode(z)
    gen_norm = gen_norm.cpu().numpy()
    gen_mv = gen_mv.cpu().numpy()
    fam_sampled = RNG.choice(real_fam, size=N_SAMPLES, replace=True)
    gen_orig = gen_norm * gen_mv[:, :, None] * fam_sampled[:, None, None]

    # Apply extinction fix (consistent with the paper's reporting)
    gen_ts = np.transpose(gen_orig, (0, 2, 1))  # (N, T, S)
    gen_ts_fixed = np.array([
        apply_extinction_threshold(gen_ts[i], threshold=0.005) for i in range(N_SAMPLES)
    ])
    gen_fixed = np.transpose(gen_ts_fixed, (0, 2, 1))  # back to (N, S, T)

    print("Computing chaos diagnostics on REAL ...")
    real_out = compute_for_population(real_orig, "real")
    print("Computing chaos diagnostics on GENERATED ...")
    gen_out = compute_for_population(gen_fixed, "gen")

    rqa_keys = ("RR", "DET", "L_mean", "L_max", "LAM", "TT")
    results = {
        "n_samples_per_group": N_SAMPLES,
        "rqa_params": {
            "embed_dim": RQA_EMBED_DIM, "embed_tau": RQA_EMBED_TAU,
            "target_recurrence_rate": RQA_RECURRENCE_RATE_TARGET,
            "min_line_length": RQA_MIN_LINE_LENGTH,
        },
        "lyap_params": {
            "embed_dim": LYAP_EMBED_DIM, "embed_tau": LYAP_EMBED_TAU,
            "theiler_window": LYAP_MEAN_PERIOD,
            "fit_fraction": LYAP_FIT_FRACTION,
        },
        "rqa": {},
        "lyap": {},
        "note": (
            "Sequences are T=65; this is a short-window regime in which Lyapunov "
            "estimation is noisy at the single-trajectory level. We report "
            "distribution-level comparisons (KS test) rather than per-trajectory "
            "precision. RQA epsilon is chosen per-trajectory to fix recurrence "
            "rate at the target value, so RR by construction matches between "
            "groups; the discriminative content lives in DET, L_mean, L_max, "
            "LAM, TT."
        ),
    }
    for k in rqa_keys:
        a = real_out["rqa"][k]
        b = gen_out["rqa"][k]
        results["rqa"][k] = {
            "real": summarize(a),
            "gen": summarize(b),
            "ks": ks_compare(a, b),
        }
    a = real_out["lyap"]
    b = gen_out["lyap"]
    results["lyap"] = {
        "real": summarize(a),
        "gen": summarize(b),
        "ks": ks_compare(a, b),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved → {OUT_JSON}")

    # --------- figure ---------
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))

    panels = [
        ("DET",   "Determinism",                axes[0, 0]),
        ("L_mean", "Mean diag. line $L_{mean}$", axes[0, 1]),
        ("L_max", "Max diag. line $L_{max}$",   axes[0, 2]),
        ("LAM",   "Laminarity",                 axes[0, 3]),
        ("TT",    "Trapping time $TT$",          axes[1, 0]),
        ("RR",    "Recurrence rate (target)",   axes[1, 1]),
    ]

    for key, label, ax in panels:
        a = real_out["rqa"][key]
        b = gen_out["rqa"][key]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        lo, hi = np.nanmin(np.r_[a, b]), np.nanmax(np.r_[a, b])
        bins = np.linspace(lo, hi, 25) if hi > lo else 10
        ax.hist(a, bins=bins, alpha=0.55, label="real", color="tab:blue", density=True)
        ax.hist(b, bins=bins, alpha=0.55, label="gen",  color="tab:orange", density=True)
        ks = results["rqa"][key]["ks"]
        ax.set_title(f"{label}\nKS={ks['ks_stat']:.3f}, p={ks['ks_p']:.1e}", fontsize=9)
        ax.set_xlabel(key)
        ax.legend(fontsize=7)

    # Lyapunov panel
    ax = axes[1, 2]
    a = real_out["lyap"][np.isfinite(real_out["lyap"])]
    b = gen_out["lyap"][np.isfinite(gen_out["lyap"])]
    lo, hi = np.nanmin(np.r_[a, b]), np.nanmax(np.r_[a, b])
    bins = np.linspace(lo, hi, 25) if hi > lo else 10
    ax.hist(a, bins=bins, alpha=0.55, label="real", color="tab:blue", density=True)
    ax.hist(b, bins=bins, alpha=0.55, label="gen", color="tab:orange", density=True)
    ks = results["lyap"]["ks"]
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"Largest Lyapunov exponent $\\lambda_1$\nKS={ks['ks_stat']:.3f}, p={ks['ks_p']:.1e}",
                 fontsize=9)
    ax.set_xlabel("$\\lambda_1$ (per timestep)")
    ax.legend(fontsize=7)

    # Summary text panel
    ax = axes[1, 3]
    ax.axis("off")
    def fmt(d):
        return f"{d['mean']:+.3f} ± {d['std']:.3f}"
    lines = [
        "Real vs generated  (mean ± std)",
        "",
    ]
    for k in rqa_keys:
        r = results["rqa"][k]
        lines.append(f"  {k:<7s} real: {fmt(r['real'])}")
        lines.append(f"  {k:<7s} gen : {fmt(r['gen'])}  KS p={r['ks']['ks_p']:.1e}")
    r = results["lyap"]
    lines.append(f"  λ₁     real: {fmt(r['real'])}")
    lines.append(f"  λ₁     gen : {fmt(r['gen'])}  KS p={r['ks']['ks_p']:.1e}")
    lines.append("")
    lines.append(f"N per group: {N_SAMPLES}")
    lines.append(f"Embedding: m={RQA_EMBED_DIM}, τ={RQA_EMBED_TAU}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8)

    fig.suptitle("Chaos diagnostics: real vs generated (n=%d each)" % N_SAMPLES,
                 fontweight="bold")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved → {OUT_FIG}")


if __name__ == "__main__":
    main()
