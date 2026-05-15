"""Produce paper figures from `RESULTS_COMPARATIVE.json`.

Inputs:
  - RESULTS_COMPARATIVE.json — output of analysis/evaluate_all_models.py
  - (optional) RESULTS_LENS_VALIDATION.json — for the lens-validation figure
  - (optional) RESULTS_CHAOS.json — v1 chaos diagnostics (for backwards-compat
    comparison if needed)

Outputs:
  - final figures/fig_comparative_table.{pdf,png}    — 7-row × N-metric heatmap
  - final figures/fig_rqa_ladder.{pdf,png}           — RQA distributions, real
                                                       vs each model
  - final figures/fig_lyapunov_distributions.{pdf,png} — λ₁ histograms
  - final figures/fig_recon_vs_chaos.{pdf,png}        — scatter: recon-R² vs
                                                       chaos-distance per model
                                                       (the headline plot)

These are skeleton figures designed to render even with partial data, so the
script can be iterated as more models complete training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical display order + labels for the comparative table
MODEL_ORDER = [
    ("cvae-scale-cond", "Scale-cond VAE"),
    ("cvae-no-scale-cond", "No-cond VAE"),
    ("cvae-stochastic", "Stochastic-decoder VAE"),
    ("latent-ode", "Latent-ODE"),
    ("transformer-vae", "Transformer-VAE"),
    ("kan-vae", "KAN-VAE"),
    ("glv-regression", "Direct GLV regression"),
]

METRIC_ORDER = [
    ("recon_R2_normalized", "Recon R² (norm)"),
    ("recon_R2_original_scale", "Recon R² (orig)"),
    ("max_val_R2_pooled", "Max-val R² (pooled)"),
    ("mmd2", "MMD² ↓"),
    ("density", "Density@5 ↑"),
    ("coverage", "Coverage@5 ↑"),
    ("DET_ks_p", "RQA-DET KS p ↑"),
    ("L_mean_ks_p", "RQA-L_mean KS p ↑"),
    ("LAM_ks_p", "RQA-LAM KS p ↑"),
    ("lyap_ks_p", "λ₁ KS p ↑"),
]


def get_metric(record, metric_key):
    """Extract a metric value from a per-model record. Returns NaN on miss."""
    rec = record.get("reconstruction", {})
    lens1 = record.get("lens1_feature_MMD", {})
    lens2 = record.get("lens2_RQA", {})
    lens3 = record.get("lens3_Lyapunov", {})

    if metric_key == "recon_R2_normalized":
        return rec.get("recon_R2_normalized", np.nan)
    if metric_key == "recon_R2_original_scale":
        return rec.get("recon_R2_original_scale", np.nan)
    if metric_key == "max_val_R2_pooled":
        return rec.get("max_val_R2_pooled", np.nan)
    if metric_key == "mmd2":
        return lens1.get("mmd", {}).get("mmd2_observed", np.nan)
    if metric_key == "density":
        return lens1.get("density_coverage", {}).get("density_at_k", np.nan)
    if metric_key == "coverage":
        return lens1.get("density_coverage", {}).get("coverage_at_k", np.nan)
    for rqa in ("RR", "DET", "L_mean", "L_max", "LAM", "TT"):
        if metric_key == f"{rqa}_ks_p":
            return lens2.get(rqa, {}).get("ks", {}).get("ks_p", np.nan)
    if metric_key == "lyap_ks_p":
        return lens3.get("ks", {}).get("ks_p", np.nan)
    return np.nan


def aggregate_by_model_type(records: dict) -> dict:
    """For multi-seed runs, aggregate to mean ± std per model type."""
    grouped = {}
    for key, rec in records.items():
        # key is "<model-type>:<checkpoint-stem>"
        model_type = key.split(":", 1)[0]
        grouped.setdefault(model_type, []).append(rec)
    out = {}
    for model_type, recs in grouped.items():
        agg = {}
        for metric_key, _ in METRIC_ORDER:
            vals = np.array([get_metric(r, metric_key) for r in recs], dtype=float)
            vals = vals[np.isfinite(vals)]
            agg[metric_key] = {
                "mean": float(vals.mean()) if vals.size else float("nan"),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "n": int(vals.size),
            }
        out[model_type] = agg
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_comparative_table(aggregated, out_path):
    """7-row × N-metric heatmap with values annotated.

    Coloring: each metric column is independently min-max scaled (or log-scaled
    for KS p-values). "Direction of better" is encoded by which end of the
    colormap is bright.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    model_types_present = [(mt, lbl) for mt, lbl in MODEL_ORDER if mt in aggregated]
    n_models = len(model_types_present)
    n_metrics = len(METRIC_ORDER)
    if n_models == 0:
        print("No models in aggregated dict — nothing to plot.")
        return
    fig, ax = plt.subplots(figsize=(1.7 + 1.0 * n_metrics, 0.6 + 0.6 * n_models))
    grid = np.full((n_models, n_metrics), np.nan)
    annotations = np.empty((n_models, n_metrics), dtype=object)
    for i, (mt, _) in enumerate(model_types_present):
        for j, (mk, _) in enumerate(METRIC_ORDER):
            v = aggregated[mt][mk]["mean"]
            s = aggregated[mt][mk]["std"]
            n = aggregated[mt][mk]["n"]
            grid[i, j] = v
            if not np.isfinite(v):
                annotations[i, j] = "—"
            elif mk.endswith("_ks_p") or mk == "mmd2":
                annotations[i, j] = f"{v:.1e}"
            else:
                annotations[i, j] = f"{v:.3f}" + (f"\n±{s:.3f}" if n > 1 else "")

    # Normalize each column independently. KS p-values: use −log10 so darker
    # = stronger distinguishability. R² and density/coverage: higher is better.
    norm_grid = np.zeros_like(grid)
    for j, (mk, _) in enumerate(METRIC_ORDER):
        col = grid[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        if mk.endswith("_ks_p"):
            # for the table: higher p = better (matches real distribution).
            # So we just min-max it.
            mn, mx = finite.min(), finite.max()
        elif mk == "mmd2":
            # smaller is better → flip
            mn, mx = finite.min(), finite.max()
            col_flip = mx - col
            col_flip_finite = col_flip[np.isfinite(col_flip)]
            ptp = col_flip_finite.max() - col_flip_finite.min() if col_flip_finite.size else 1.0
            norm_grid[:, j] = (col_flip - col_flip_finite.min()) / max(ptp, 1e-12)
            continue
        else:
            mn, mx = finite.min(), finite.max()
        norm_grid[:, j] = (col - mn) / max(mx - mn, 1e-12)

    im = ax.imshow(norm_grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([lbl for _, lbl in METRIC_ORDER], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([lbl for _, lbl in model_types_present])
    for i in range(n_models):
        for j in range(n_metrics):
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=7,
                    color="black" if norm_grid[i, j] > 0.3 else "white")
    ax.set_title("Comparative evaluation across architectures  (each column rank-normalized)",
                 fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved → {out_path}")


def fig_chaos_summary(aggregated, out_path):
    """Side-by-side: real vs each model on DET, λ₁."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    model_types_present = [(mt, lbl) for mt, lbl in MODEL_ORDER if mt in aggregated]
    if not model_types_present:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics = [("DET", "RQA-DET (higher = more deterministic)"),
               ("lyap", "λ₁ (higher = more chaotic)")]
    # We need per-model mean/std of the *real* and *gen* values, not the KS p.
    # Right now these aren't in `aggregated` (which only tracks aggregate
    # metric scalars). The records themselves have the raw values — but
    # aggregation across seeds will be folded into a future version.
    # For now: just put a placeholder note in the figure.
    for ax, (k, title) in zip(axes, metrics):
        ax.set_title(title)
        ax.text(0.5, 0.5,
                "(per-architecture chaos-metric distributions will plot here\n"
                "once we have multi-seed RESULTS_COMPARATIVE.json with the\n"
                "raw per-sample arrays — currently the JSON stores only\n"
                "aggregated means.)",
                ha="center", va="center", fontsize=9, family="monospace",
                transform=ax.transAxes)
        ax.axis("off")
    fig.suptitle("Chaos diagnostics by architecture (placeholder)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_recon_vs_chaos(aggregated, out_path):
    """Scatter: x = recon R² (normalized), y = chaos-mismatch (KS p inverted).

    The headline plot. Architectures that win on recon may lose on dynamical
    fidelity (top-right corner = ideal: high recon AND high p). Architectures
    in the bottom-right (high recon, low p) are the "regress-to-mean" failure
    mode.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    model_types_present = [(mt, lbl) for mt, lbl in MODEL_ORDER if mt in aggregated]
    if not model_types_present:
        return
    fig, ax = plt.subplots(figsize=(7, 5))

    xs, ys, labels = [], [], []
    for mt, lbl in model_types_present:
        rec = aggregated[mt]
        x = rec.get("recon_R2_original_scale", {}).get("mean", np.nan)
        # y: take the geometric mean of (1 - log10(p)) across the four NLD KS p-values
        ps = []
        for mk in ("DET_ks_p", "L_mean_ks_p", "LAM_ks_p", "lyap_ks_p"):
            p = rec.get(mk, {}).get("mean", np.nan)
            if np.isfinite(p) and p > 0:
                ps.append(p)
        if not ps:
            continue
        y = float(np.exp(np.mean(np.log(ps))))   # geometric mean of p
        xs.append(x); ys.append(y); labels.append(lbl)

    if not xs:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
    else:
        sc = ax.scatter(xs, ys, s=120, c=range(len(xs)), cmap="tab10", edgecolor="black")
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Reconstruction $R^2$ (original scale; higher = better recon)")
        ax.set_ylabel("Geometric mean of NLD KS p-values\n(higher = better dynamical match)")
        ax.set_yscale("log")
        ax.axhline(0.05, color="red", ls="--", lw=0.7, label="p = 0.05")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Recon quality vs. dynamical fidelity\n"
                     "(top-right = recon-fidelity Pareto frontier)",
                     fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="RESULTS_COMPARATIVE.json")
    ap.add_argument("--out-dir", default="final figures")
    args = ap.parse_args()

    p = Path(args.results)
    if not p.exists():
        print(f"WARNING: {p} does not exist yet — using sanity JSON for layout test.")
        p = Path("RESULTS_COMPARATIVE_v1_sanity.json")
        if not p.exists():
            print("No comparative data on disk; aborting.")
            return
    records = json.loads(p.read_text())
    aggregated = aggregate_by_model_type(records)
    print(f"Found {len(aggregated)} model types in {p}:")
    for mt in aggregated:
        print(f"  - {mt}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_comparative_table(aggregated, out_dir / "fig_comparative_table.pdf")
    fig_chaos_summary(aggregated, out_dir / "fig_chaos_summary.pdf")
    fig_recon_vs_chaos(aggregated, out_dir / "fig_recon_vs_chaos.pdf")


if __name__ == "__main__":
    main()
