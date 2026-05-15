"""Lens-validation experiment (§5 of the pivot design doc).

The 3-lens protocol (feature-MMD + RQA + Rosenstein λ₁) claims to detect
dynamical-fidelity defects that recon-R² misses. This script *demonstrates*
that claim on controlled synthetic perturbations:

  (a) low-pass filter (smoothing)            — should hit RQA and λ₁
  (b) high-frequency noise                    — should hit RQA and λ₁
  (c) amplitude rescaling                     — should be invisible to the lenses
                                                (they work on normalized signals
                                                 or on per-trajectory recurrence
                                                 rate); detectable only by
                                                 scale-aware metrics
  (d) phase shift / time-warp                — should hit RQA structure
  (e) species permutation                    — should hit feature-MMD on
                                                per-species features but NOT
                                                on global features

For each perturbation we measure:
  - recon-R² (perturbed vs. original, in normalized space)
  - feature-MMD permutation p-value vs. unperturbed
  - RQA KS p-values (5 measures)
  - Lyapunov KS p-value

Output: RESULTS_LENS_VALIDATION.json + final figures/fig_lens_validation.pdf.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.chaos_diagnostics import rqa_measures, rosenstein_lyapunov  # noqa: E402
from analysis.evaluate_all_models import (  # noqa: E402
    featurize, mmd_permutation_test, DROPPED_FEATURES,
)

REAL_PATH = REPO_ROOT / "data" / "TEST_FINAL_NOSORT.pkl"
OUT_JSON = REPO_ROOT / "RESULTS_LENS_VALIDATION.json"
OUT_FIG = REPO_ROOT / "final figures" / "fig_lens_validation.pdf"

N = 200
RNG = np.random.default_rng(2026_05_15)


# ---------------------------------------------------------------------------
# Perturbations (each takes (N, 7, T) and returns (N, 7, T))
# ---------------------------------------------------------------------------

def perturb_lowpass(X, sigma_t=2.0):
    """Gaussian smoothing along the time axis."""
    return gaussian_filter1d(X, sigma=sigma_t, axis=2, mode="nearest")


def perturb_hf_noise(X, sigma=0.05):
    return X + RNG.normal(0, sigma, size=X.shape)


def perturb_amplitude(X, factor_range=(0.5, 2.0)):
    factors = RNG.uniform(factor_range[0], factor_range[1], size=(X.shape[0], 1, 1))
    return X * factors


def perturb_phase_shift(X, max_shift=5):
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        s = int(RNG.integers(1, max_shift + 1))
        out[i] = np.roll(X[i], shift=s, axis=1)
    return out


def perturb_species_permutation(X):
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        perm = RNG.permutation(X.shape[1])
        out[i] = X[i, perm]
    return out


PERTURBATIONS = {
    "(a) low-pass filter (σ_t=2)": perturb_lowpass,
    "(b) HF noise (σ=0.05)": perturb_hf_noise,
    "(c) amplitude rescale [0.5, 2.0]": perturb_amplitude,
    "(d) phase shift (≤5 steps)": perturb_phase_shift,
    "(e) species permutation": perturb_species_permutation,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recon_R2(X_orig, X_perturbed):
    ss_res = ((X_orig - X_perturbed) ** 2).sum()
    ss_tot = ((X_orig - X_orig.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def chaos_metrics(trajs):
    """Compute RQA + Lyapunov on each species-averaged signal."""
    rqa_keys = ("RR", "DET", "L_mean", "L_max", "LAM", "TT")
    rqa = {k: [] for k in rqa_keys}
    lyap = []
    for traj in trajs:
        sig = traj.mean(axis=0)
        m = rqa_measures(sig)
        for k in rqa_keys:
            rqa[k].append(m[k])
        lyap.append(rosenstein_lyapunov(sig))
    return ({k: np.array(v, dtype=float) for k, v in rqa.items()},
            np.array(lyap, dtype=float))


def feature_mmd(real, perturbed):
    F_r, labels = featurize(real)
    F_p, _ = featurize(perturbed)
    keep = [i for i, l in enumerate(labels) if l not in DROPPED_FEATURES]
    F_r = F_r[:, keep]
    F_p = F_p[:, keep]
    sc = StandardScaler().fit(np.vstack([F_r, F_p]))
    return mmd_permutation_test(sc.transform(F_r), sc.transform(F_p),
                                n_perm=200, rng=RNG)


def ks_p(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    _, p = stats.ks_2samp(a, b)
    return float(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading real trajectories...")
    with open(REAL_PATH, "rb") as f:
        d = pickle.load(f)
    X_norm = d["data"]
    mv = d["reconstruction_max_values"]
    fam = d["family_max_values"]

    idx = RNG.choice(X_norm.shape[0], size=N, replace=False)
    real = X_norm[idx] * mv[idx][:, :, None] * fam[idx][:, None, None]  # (N, 7, T)
    print(f"  Real samples: {real.shape}")

    print("Baseline chaos metrics on real (no perturbation)...")
    rqa_real, lyap_real = chaos_metrics(real)

    results = {
        "N_per_group": N,
        "perturbations": {},
        "real_chaos_summary": {
            **{f"{k}_mean": float(np.nanmean(rqa_real[k])) for k in rqa_real},
            "lyap_mean": float(np.nanmean(lyap_real)),
        },
    }

    for name, pert in PERTURBATIONS.items():
        print(f"\n--- Perturbation: {name} ---")
        pert_X = pert(real)

        r2 = recon_R2(real, pert_X)
        print(f"  recon R² (orig vs perturbed): {r2:.4f}")

        # Lens 1
        mmd = feature_mmd(real, pert_X)
        print(f"  Lens 1 MMD² = {mmd['mmd2_observed']:.4f}, p = {mmd['p_value_permutation']:.4f}")

        # Lens 2 + 3
        rqa_p, lyap_p = chaos_metrics(pert_X)
        rqa_ks = {k: ks_p(rqa_real[k], rqa_p[k]) for k in rqa_p}
        lyap_ks = ks_p(lyap_real, lyap_p)
        print(f"  Lens 2 RQA KS p-values: " +
              ", ".join([f"{k}={rqa_ks[k]:.1e}" for k in ("DET", "L_mean", "LAM")]))
        print(f"  Lens 3 λ₁ KS p-value:    {lyap_ks:.1e}")

        results["perturbations"][name] = {
            "recon_R2": r2,
            "lens1_mmd2": mmd["mmd2_observed"],
            "lens1_mmd_p": mmd["p_value_permutation"],
            "lens2_rqa_ks": rqa_ks,
            "lens3_lyap_ks": lyap_ks,
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_JSON}")

    # ---- figure ----
    pert_names = list(PERTURBATIONS.keys())
    n_pert = len(pert_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: recon-R² bar plot
    ax = axes[0]
    r2s = [results["perturbations"][p]["recon_R2"] for p in pert_names]
    ax.barh(pert_names, r2s, color="tab:gray")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("recon $R^2$  (orig vs. perturbed; higher = harder to detect by recon alone)")
    ax.axvline(0.9, color="tab:green", linestyle="--", lw=0.8, label="0.9 (often the threshold for 'good')")
    ax.legend(fontsize=8)
    ax.set_title("(a) Recon $R^2$ alone — many perturbations sneak past")

    # Right: lens heatmap (rows = perturbation, cols = lens metrics)
    ax = axes[1]
    metric_cols = ["MMD p", "DET p", "L_mean p", "LAM p", "λ₁ p"]
    grid = []
    for p in pert_names:
        row = [
            results["perturbations"][p]["lens1_mmd_p"],
            results["perturbations"][p]["lens2_rqa_ks"]["DET"],
            results["perturbations"][p]["lens2_rqa_ks"]["L_mean"],
            results["perturbations"][p]["lens2_rqa_ks"]["LAM"],
            results["perturbations"][p]["lens3_lyap_ks"],
        ]
        grid.append(row)
    grid = np.array(grid)
    # Plot -log10(p) so darker = stronger detection
    with np.errstate(divide="ignore"):
        log_grid = -np.log10(np.clip(grid, 1e-300, 1.0))
    im = ax.imshow(log_grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right")
    ax.set_yticks(range(n_pert))
    ax.set_yticklabels(pert_names)
    ax.set_title("(b) 3-lens p-values  ($-\\log_{10} p$;  darker = stronger detection)")
    # Annotate cells with the p-value
    for i in range(n_pert):
        for j in range(len(metric_cols)):
            ax.text(j, i, f"{grid[i, j]:.0e}", ha="center", va="center",
                    fontsize=7, color="white" if log_grid[i, j] > 5 else "black")
    plt.colorbar(im, ax=ax, label="$-\\log_{10} p$")

    fig.suptitle("Lens validation: detection sensitivity vs. perturbation type", fontweight="bold")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved → {OUT_FIG}")


if __name__ == "__main__":
    main()
