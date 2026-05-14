"""
Novelty / coverage analysis for the paper.

Goal: answer the question "is the generative model just memorizing the
training distribution?" with something stronger than nearest-neighbor
distance (which already showed memorization ratio ≈ 0.95, a yellow flag).

Approach
--------
1. Build a feature representation for trajectories that captures dynamical
   properties (the same family of features used in
   `investigate_latent_interpretability.py`): per-species trend, variance,
   oscillation power, plus global mean trend/correlation/extrema/curvature.
2. Compute the **two-sample MMD test** (Gaussian kernel) between two
   "generated vs held-out test" feature distributions, plus a permutation-
   test p-value. Null: distributions are identical.
3. Also compute **density-coverage** (Naeem et al., 2020) — precision /
   recall scores for generative models:
   - **Density**: average over real points of (#generated neighbors within
     k-NN radius of each real)/k.
   - **Coverage**: fraction of real points whose k-NN ball contains at
     least one generated sample.
4. Single PDF figure summarizing the results, JSON for numbers.

This is a textbook tooling step for generative model evaluation; reviewers
will recognize it and stop asking "but is it memorizing?".
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.cvae import LSTM_VAE  # noqa: E402
from utils.post_processing import apply_extinction_threshold  # noqa: E402

MODEL_PATH = REPO_ROOT / "model_ckpts" / "model_final_30_conditioned.pth"
TEST_PATH = REPO_ROOT / "data" / "TEST_FINAL_PROCESSED.pkl"

OUT_JSON = REPO_ROOT / "RESULTS_NOVELTY.json"
OUT_FIG = REPO_ROOT / "final figures" / "fig_novelty_coverage.pdf"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 2000   # per group (real and generated)
K_NN = 5           # k for density/coverage
RNG = np.random.default_rng(42)

MODEL_CONFIG = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": "log",
    "use_scale_conditioning": True,
}


# ---------------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------------


def features_for_one(traj: np.ndarray) -> np.ndarray:
    """traj shape (7, 65). Returns 1D feature vector.

    Features:
      - per-species (7 species × 3 = 21):
          mean, std, linear trend slope
      - global (5):
          total variance, mean pairwise correlation,
          mean extrema count, mean curvature, mean dominant frequency.
    """
    S, T = traj.shape
    feats = []

    t = np.arange(T)
    for s in range(S):
        x = traj[s]
        feats.append(x.mean())
        feats.append(x.std())
        # linear trend
        slope, _ = np.polyfit(t, x, 1)
        feats.append(slope)

    # global features
    feats.append(float(traj.std(axis=1).sum()))  # total variance proxy

    # mean pairwise correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = np.corrcoef(traj)
    iu = np.triu_indices(S, k=1)
    feats.append(float(np.nanmean(C[iu])))

    # mean extrema count
    def extrema_count(x):
        d = np.diff(x)
        return int(((d[:-1] * d[1:]) < 0).sum())
    feats.append(float(np.mean([extrema_count(traj[s]) for s in range(S)])))

    # mean curvature (mean |2nd diff|)
    feats.append(float(np.mean(np.abs(np.diff(traj, n=2, axis=1)))))

    # mean dominant frequency
    def dominant_freq(x):
        spec = np.abs(np.fft.rfft(x - x.mean()))
        if spec.sum() == 0:
            return 0.0
        return float(np.argmax(spec[1:]) + 1) / len(x)
    feats.append(float(np.mean([dominant_freq(traj[s]) for s in range(S)])))

    return np.array(feats, dtype=np.float64)


def featurize(trajs: np.ndarray) -> np.ndarray:
    """trajs (N, 7, 65) → (N, F)."""
    return np.stack([features_for_one(trajs[i]) for i in range(trajs.shape[0])], axis=0)


# ---------------------------------------------------------------------------
# 2. MMD with Gaussian kernel + permutation test
# ---------------------------------------------------------------------------


def median_heuristic_sigma(X: np.ndarray) -> float:
    """Median pairwise distance as kernel bandwidth (standard heuristic)."""
    n = min(500, X.shape[0])
    sub = X[RNG.choice(X.shape[0], size=n, replace=False)]
    d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
    d = d[np.triu_indices(n, k=1)]
    return float(np.median(d))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD^2 estimate with Gaussian kernel exp(-||x-y||^2 / (2 σ^2))."""
    def K(A, B):
        d = np.linalg.norm(A[:, None] - B[None, :], axis=-1) ** 2
        return np.exp(-d / (2 * sigma ** 2))

    Kxx = K(X, X)
    Kyy = K(Y, Y)
    Kxy = K(X, Y)
    n = X.shape[0]
    m = Y.shape[0]
    # unbiased: zero out diagonals of Kxx, Kyy
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2 * Kxy.sum() / (n * m)
    )


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray, n_perm: int = 500) -> dict:
    sigma = median_heuristic_sigma(np.vstack([X, Y]))
    observed = mmd2_unbiased(X, Y, sigma)
    nx = X.shape[0]
    combined = np.vstack([X, Y])
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = RNG.permutation(combined.shape[0])
        perm_stats[i] = mmd2_unbiased(combined[idx[:nx]], combined[idx[nx:]], sigma)
    p = float((perm_stats >= observed).sum() + 1) / (n_perm + 1)
    return {
        "sigma": sigma,
        "mmd2_observed": observed,
        "p_value_permutation": p,
        "n_permutations": n_perm,
        "perm_stats_mean": float(perm_stats.mean()),
        "perm_stats_std": float(perm_stats.std()),
    }


# ---------------------------------------------------------------------------
# 3. Density-coverage (Naeem et al. 2020)
# ---------------------------------------------------------------------------


def density_coverage(real: np.ndarray, gen: np.ndarray, k: int = 5) -> dict:
    """Compute Naeem-Lee precision/recall variants.

    - density = (1/k) * (1/M) * sum_j sum_i 1(g_j ∈ B(r_i, NN_k(r_i)))
    - coverage = (1/N) * sum_i 1(∃ j with g_j ∈ B(r_i, NN_k(r_i)))
    """
    # Fit k-NN on real to get radii
    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real)
    dist_real, _ = nn_real.kneighbors(real)
    radii = dist_real[:, -1]  # distance to k-th neighbor (excluding self at idx 0 → idx k)

    # For each real point r_i, count how many gen points fall within radii[i]
    nn_to_gen = NearestNeighbors().fit(gen)
    counts = np.zeros(real.shape[0], dtype=np.int32)
    for i, r_i in enumerate(real):
        in_ball = nn_to_gen.radius_neighbors(r_i.reshape(1, -1), radius=radii[i],
                                             return_distance=False)
        counts[i] = len(in_ball[0])

    density = float(counts.sum() / (k * real.shape[0]))
    coverage = float((counts > 0).mean())
    return {"density_at_k": density, "coverage_at_k": coverage, "k": k}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main():
    # Load model + test data
    print("Loading model & test data...")
    model = LSTM_VAE(MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with open(TEST_PATH, "rb") as f:
        d = pickle.load(f)
    real_norm = d["data"]
    real_mv = d["reconstruction_max_values"]
    real_fam = d["family_max_values"]

    # Real, denormalized to original scale
    real_idx = RNG.choice(real_norm.shape[0], size=N_SAMPLES, replace=False)
    real_orig = (
        real_norm[real_idx]
        * real_mv[real_idx][:, :, None]
        * real_fam[real_idx][:, None, None]
    )

    # Generated, denormalized to original scale with family_max sampled from data
    print(f"Generating {N_SAMPLES} samples from the trained model...")
    torch.manual_seed(123)
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

    print("Computing dynamical feature vectors...")
    F_real = featurize(real_orig)
    F_gen = featurize(gen_fixed)
    print(f"  Feature dim: {F_real.shape[1]}")

    # Standardize features (Z-score on the pooled distribution) so MMD/density
    # are not dominated by the largest-scale dimensions.
    scaler = StandardScaler().fit(np.vstack([F_real, F_gen]))
    F_real_z = scaler.transform(F_real)
    F_gen_z = scaler.transform(F_gen)

    print("Running MMD permutation test...")
    mmd = mmd_permutation_test(F_real_z, F_gen_z, n_perm=500)
    print(f"  MMD² (observed)  = {mmd['mmd2_observed']:.6f}")
    print(f"  permutation null = {mmd['perm_stats_mean']:.6f} ± {mmd['perm_stats_std']:.6f}")
    print(f"  p-value          = {mmd['p_value_permutation']:.4f}")

    print("Computing density / coverage...")
    dc = density_coverage(F_real_z, F_gen_z, k=K_NN)
    print(f"  density@k={K_NN}  = {dc['density_at_k']:.3f}")
    print(f"  coverage@k={K_NN} = {dc['coverage_at_k']:.3f}")

    # Also compute symmetric version for additional context: density(gen, real)
    dc_swap = density_coverage(F_gen_z, F_real_z, k=K_NN)
    print(f"  density(swap) gen→real = {dc_swap['density_at_k']:.3f}")
    print(f"  coverage(swap)         = {dc_swap['coverage_at_k']:.3f}")

    # Per-feature 1D KS tests (which features distinguish real from generated?)
    print("Per-feature KS tests...")
    ks_results = []
    for j in range(F_real_z.shape[1]):
        ks_stat, ks_p = stats.ks_2samp(F_real_z[:, j], F_gen_z[:, j])
        ks_results.append({"feature_idx": j, "ks_stat": float(ks_stat),
                           "ks_p": float(ks_p)})
    n_sig = sum(1 for r in ks_results if r["ks_p"] < 0.05)
    print(f"  {n_sig}/{len(ks_results)} features differ at p<0.05")

    # Per-species + global feature labels
    feature_labels = []
    for s in range(7):
        for stat in ("mean", "std", "trend"):
            feature_labels.append(f"sp{s}_{stat}")
    feature_labels += ["total_var", "mean_corr", "mean_extrema",
                       "mean_curv", "mean_freq"]

    results = {
        "n_samples_per_group": N_SAMPLES,
        "feature_dim": int(F_real_z.shape[1]),
        "feature_labels": feature_labels,
        "mmd": mmd,
        "density_coverage": dc,
        "density_coverage_swapped": dc_swap,
        "ks_per_feature": ks_results,
        "ks_n_significant_at_p05": n_sig,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved → {OUT_JSON}")

    # ----- figure -----
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # A) MMD permutation histogram
    ax = axes[0, 0]
    ax.hist(np.random.default_rng(0).permutation([])  # placeholder; we plot below
            , bins=30)
    ax.clear()
    # rerun small permutation set just for the figure (use the JSON stats directly)
    mean_null = mmd["perm_stats_mean"]
    std_null = mmd["perm_stats_std"]
    xs = np.linspace(mean_null - 4 * std_null,
                     max(mean_null + 4 * std_null, mmd["mmd2_observed"] * 1.05), 200)
    ax.plot(xs, stats.norm.pdf(xs, mean_null, std_null),
            label="permutation null (Gaussian approx)")
    ax.axvline(mmd["mmd2_observed"], color="red", lw=2,
               label=f"observed MMD² = {mmd['mmd2_observed']:.4f}")
    ax.axvline(mean_null, color="gray", ls="--",
               label=f"null mean = {mean_null:.4f}")
    ax.set_xlabel("MMD² (Gaussian kernel)")
    ax.set_ylabel("density")
    ax.set_title(f"A) MMD permutation test  —  p = {mmd['p_value_permutation']:.4f}")
    ax.legend(fontsize=7, loc="upper left")

    # B) Density / coverage bars
    ax = axes[0, 1]
    metrics = ["density@k", "coverage@k", "density (gen→real)", "coverage (swap)"]
    vals = [dc["density_at_k"], dc["coverage_at_k"],
            dc_swap["density_at_k"], dc_swap["coverage_at_k"]]
    colors = ["tab:green" if v >= 0.8 else "tab:orange" if v >= 0.5 else "tab:red" for v in vals]
    ax.barh(metrics, vals, color=colors)
    ax.axvline(1.0, color="k", lw=0.5, ls="--")
    ax.set_xlim(0, max(1.05, max(vals) * 1.1))
    ax.set_title(f"B) Density / coverage  (k={K_NN})")

    # C) Per-feature KS test bar
    ax = axes[1, 0]
    ks_stats = [r["ks_stat"] for r in ks_results]
    ks_ps = [r["ks_p"] for r in ks_results]
    colors = ["tab:red" if p < 0.05 else "tab:gray" for p in ks_ps]
    ax.bar(range(len(ks_stats)), ks_stats, color=colors)
    ax.set_xticks(range(len(ks_stats)))
    ax.set_xticklabels(feature_labels, rotation=90, fontsize=6)
    ax.set_ylabel("KS statistic")
    ax.set_title(f"C) Per-feature KS test  ({n_sig}/{len(ks_stats)} significant p<0.05)")
    ax.axhline(0, color="k", lw=0.5)

    # D) Verdict text
    ax = axes[1, 1]
    ax.axis("off")
    if mmd["p_value_permutation"] < 0.01:
        verdict = "Distributions DIFFER significantly (p < 0.01)."
    elif mmd["p_value_permutation"] < 0.05:
        verdict = "Marginal difference (p < 0.05)."
    else:
        verdict = "No significant difference — generated and real are\nstatistically indistinguishable in feature space."

    lines = [
        "VERDICT",
        "",
        verdict,
        "",
        f"  N per group   : {N_SAMPLES}",
        f"  feature dim   : {F_real_z.shape[1]}",
        f"  MMD² obs / null mean : {mmd['mmd2_observed']:.4f} / {mean_null:.4f}",
        f"  MMD permutation p    : {mmd['p_value_permutation']:.4f}",
        "",
        f"  density@{K_NN}  : {dc['density_at_k']:.3f}   (≥1 = generated covers real well)",
        f"  coverage@{K_NN} : {dc['coverage_at_k']:.3f}   (1 = every real has a gen neighbor)",
        "",
        f"  features distinguishing real vs gen: {n_sig}/{len(ks_results)}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)

    fig.suptitle("Novelty / coverage analysis — generated vs held-out test",
                 fontweight="bold")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved → {OUT_FIG}")


if __name__ == "__main__":
    main()
