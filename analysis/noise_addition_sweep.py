"""Noise-addition sweep — Tier 1.2 of the post-audit pipeline.

For each VAE-family checkpoint we have, generate 200 clean samples, apply
lognormal multiplicative noise at σ ∈ {0, 0.005, 0.01, 0.02, 0.05},
and compute Lens 2 (RQA-DET) + Lens 3 (Lyapunov) + Lens 1 (MMD) vs real
test data.

The centerpiece paper figure for the noise-modeling story (PROJECT.md §1.3.0):
demonstrate that adding back the σ=0.01 lognormal noise the test data was
generated with closes the protocol-detected gap to non-significance.

Output:
  - RESULTS_NOISE_SWEEP.json  : per-model, per-σ, per-distribution KS p-values
  - final figures/fig_noise_sweep.{pdf,png} : the headline figure
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.chaos_diagnostics import rqa_measures, rosenstein_lyapunov  # noqa: E402
from analysis.evaluate_all_models import ADAPTERS, featurize, mmd_permutation_test, DROPPED_FEATURES  # noqa: E402
from utils.post_processing import apply_extinction_threshold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GEN = 200
N_REAL = 200
RNG_SEED = 2026_05_19
SIGMA_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.05]

# Architecture → (eval-harness key, list of seeds available, b1 cure variants)
# We pull whatever ckpts exist on disk at run time.

OUT_JSON = REPO_ROOT / "RESULTS_NOISE_SWEEP.json"
OUT_FIG = REPO_ROOT / "final figures" / "fig_noise_sweep.pdf"


def add_lognormal_noise(X, sigma, rng):
    if sigma <= 0:
        return X
    scaling = float(np.exp(sigma ** 2 / 2))
    noise = rng.lognormal(0, sigma, size=X.shape)
    return X * (noise / scaling)


def chaos_dist(trajs):
    rqa_dets, lyaps = [], []
    for traj in trajs:
        sig = traj.mean(axis=0)
        rqa_dets.append(rqa_measures(sig)["DET"])
        lyaps.append(rosenstein_lyapunov(sig))
    return np.array(rqa_dets, dtype=float), np.array(lyaps, dtype=float)


def compute_mmd_against_real(F_real_z, gen_trajs, scaler, rng):
    F_gen, labels = featurize(gen_trajs)
    keep = [i for i, l in enumerate(labels) if l not in DROPPED_FEATURES]
    F_gen = F_gen[:, keep]
    F_gen_z = scaler.transform(F_gen)
    return mmd_permutation_test(F_real_z, F_gen_z, n_perm=100, rng=rng)


def discover_checkpoints():
    """Find all VAE-family ckpts on disk (m1-m6 with any seed)."""
    ckpts = []
    type_map = {
        1: "cvae-scale-cond", 2: "cvae-no-scale-cond", 3: "cvae-stochastic",
        4: "latent-ode", 5: "transformer-vae", 6: "kan-vae",
    }
    for m, mtype in type_map.items():
        for seed in [42, 123, 2026]:
            p = REPO_ROOT / "model_ckpts" / f"model_{m}_seed{seed}.pth"
            if p.exists():
                ckpts.append((mtype, str(p), f"m{m}_seed{seed}"))
    return ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-path", default="data/TEST_FINAL_NOSORT.pkl")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    args = ap.parse_args()

    print(f"Loading test set from {args.test_path}...")
    with open(args.test_path, "rb") as f:
        test_pkg = pickle.load(f)
    N_test = test_pkg["data"].shape[0]

    rng = np.random.default_rng(RNG_SEED)
    real_idx = rng.choice(N_test, size=N_REAL, replace=False)
    real_orig = (
        test_pkg["data"][real_idx]
        * test_pkg["reconstruction_max_values"][real_idx][:, :, None]
        * test_pkg["family_max_values"][real_idx][:, None, None]
    )

    # Real chaos diagnostics (once, all gens are compared to this)
    print("Computing real-data chaos diagnostics...")
    det_real, lyap_real = chaos_dist(real_orig)
    print(f"  Real DET: {det_real.mean():.4f} ± {det_real.std():.4f}")
    print(f"  Real λ₁:  {np.nanmean(lyap_real):+.4f} ± {np.nanstd(lyap_real):.4f}")

    # Real-feature setup for MMD reuse across all σ/model evaluations
    F_real, feat_labels = featurize(real_orig)
    keep_idx = [i for i, l in enumerate(feat_labels) if l not in DROPPED_FEATURES]
    F_real = F_real[:, keep_idx]
    scaler = StandardScaler().fit(F_real)
    F_real_z = scaler.transform(F_real)

    # Find all ckpts to process
    ckpts = discover_checkpoints()
    print(f"\nFound {len(ckpts)} ckpts:")
    for _, _, label in ckpts:
        print(f"  {label}")

    results = {
        "n_real": N_REAL, "n_gen": args.n_gen, "test_path": args.test_path,
        "real_DET_mean": float(det_real.mean()), "real_DET_std": float(det_real.std()),
        "real_lyap_mean": float(np.nanmean(lyap_real)),
        "real_lyap_std": float(np.nanstd(lyap_real)),
        "sigma_sweep": SIGMA_SWEEP,
        "models": {},
    }

    for mtype, path, label in ckpts:
        print(f"\n=== {label} ({mtype}) ===")
        try:
            model = ADAPTERS[mtype](Path(path), DEVICE)
        except Exception as e:
            print(f"  LOAD ERR: {e}")
            continue

        # Generate once (clean), apply different σ noise post-hoc
        torch.manual_seed(20260519)
        with torch.no_grad():
            X_gen, gen_mv = model.generate(args.n_gen, DEVICE)
        X_gen_np = X_gen.cpu().numpy()
        gen_mv_np = gen_mv.cpu().numpy()

        # Denormalize
        fam_sampled = rng.choice(test_pkg["family_max_values"],
                                 size=args.n_gen, replace=True)
        gen_orig_clean = X_gen_np * gen_mv_np[:, :, None] * fam_sampled[:, None, None]

        # Apply extinction-fix (consistent with eval harness)
        gen_ts = np.transpose(gen_orig_clean, (0, 2, 1))
        gen_ts_fixed = np.array([
            apply_extinction_threshold(gen_ts[i], threshold=0.005)
            for i in range(args.n_gen)
        ])
        gen_clean = np.transpose(gen_ts_fixed, (0, 2, 1))

        model_results = []
        for sigma in SIGMA_SWEEP:
            print(f"  σ = {sigma:>5.3f}")
            noise_rng = np.random.default_rng(RNG_SEED + int(sigma * 1000))
            gen_noisy = add_lognormal_noise(gen_clean, sigma, noise_rng)
            det_g, lyap_g = chaos_dist(gen_noisy)

            ks_det = stats.ks_2samp(det_real, det_g)
            lyap_g_finite = lyap_g[np.isfinite(lyap_g)]
            lyap_r_finite = lyap_real[np.isfinite(lyap_real)]
            ks_lyap = stats.ks_2samp(lyap_r_finite, lyap_g_finite) if (
                len(lyap_g_finite) > 5 and len(lyap_r_finite) > 5
            ) else None

            mmd = compute_mmd_against_real(F_real_z, gen_noisy, scaler, noise_rng)

            entry = {
                "sigma": sigma,
                "gen_DET_mean": float(det_g.mean()),
                "gen_DET_std": float(det_g.std()),
                "det_KS_stat": float(ks_det.statistic),
                "det_KS_p": float(ks_det.pvalue),
                "gen_lyap_mean": float(np.nanmean(lyap_g)),
                "gen_lyap_std": float(np.nanstd(lyap_g)),
                "lyap_KS_p": float(ks_lyap.pvalue) if ks_lyap is not None else None,
                "mmd2": float(mmd["mmd2_observed"]),
                "mmd_p": float(mmd["p_value_permutation"]),
            }
            model_results.append(entry)
            print(f"    DET = {det_g.mean():.3f} ± {det_g.std():.3f}  "
                  f"(KS p {ks_det.pvalue:.2e}); "
                  f"λ₁ = {np.nanmean(lyap_g):+.4f} (KS p {entry['lyap_KS_p']:.2e}); "
                  f"MMD² = {mmd['mmd2_observed']:.2e}")

        results["models"][label] = {"type": mtype, "sweep": model_results}

        # Write incrementally
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → {args.out} updated")

    # Make headline figure: x = σ, dual y = DET KS p / λ₁ KS p, one line per model
    print("\nBuilding fig_noise_sweep...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in [(axes[0], "det_KS_p", "RQA-DET KS p vs real"),
                           (axes[1], "lyap_KS_p", "Lyapunov λ₁ KS p vs real")]:
        for label, m in results["models"].items():
            sigmas = [e["sigma"] for e in m["sweep"]]
            ps = [e[key] if e[key] is not None else float("nan") for e in m["sweep"]]
            ax.semilogy(sigmas, ps, marker="o", label=label, alpha=0.7)
        ax.axhline(0.05, color="r", linestyle="--", lw=0.8, label="p = 0.05")
        ax.set_xlabel("σ (lognormal noise added to clean VAE output)")
        ax.set_ylabel("KS p-value vs real test data")
        ax.set_title(title)
        ax.set_ylim(1e-40, 2)
        ax.legend(fontsize=6, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Noise-addition sweep: do generators match real after adding back the observation noise?",
                 fontweight="bold")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"  → {OUT_FIG}")


if __name__ == "__main__":
    main()
