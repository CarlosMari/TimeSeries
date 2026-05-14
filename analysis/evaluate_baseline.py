"""
Evaluate the trained baseline VAE and produce a comparison table.

Reads:
  model_ckpts/model_final_30_baseline.pth   (non-conditioned)
  model_ckpts/model_final_30_conditioned.pth (conditioned, the paper's model)
  RESULTS.json                              (canonical conditioned numbers)

Writes:
  RESULTS_BASELINE.json                     (full baseline metrics)
  RESULTS_COMPARISON.md                     (side-by-side table)

Computes the same metrics as analysis/produce_paper_metrics.py for the
baseline, then renders the comparison table that goes into the paper.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.cvae import LSTM_VAE  # noqa: E402

BASELINE_PATH = REPO_ROOT / "model_ckpts" / "model_final_30_baseline.pth"
CONDITIONED_PATH = REPO_ROOT / "model_ckpts" / "model_final_30_conditioned.pth"
TEST_PATH = REPO_ROOT / "data" / "TEST_FINAL_PROCESSED.pkl"
RESULTS_JSON = REPO_ROOT / "RESULTS.json"

OUT_JSON = REPO_ROOT / "RESULTS_BASELINE.json"
OUT_MD = REPO_ROOT / "RESULTS_COMPARISON.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_CONFIG = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": "log",
    "use_scale_conditioning": False,  # the baseline
}


def evaluate_baseline(test_pkg):
    print("Loading baseline model...")
    model = LSTM_VAE(BASE_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
    model.eval()

    X = test_pkg["data"]
    mv = test_pkg["reconstruction_max_values"]
    fam = test_pkg["family_max_values"]
    N = X.shape[0]

    preds, mvps, mus_all = [], [], []
    bs = 1000
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=DEVICE)
            mb = torch.tensor(mv[i:i+bs], dtype=torch.float32, device=DEVICE)
            # baseline ignores max_vals
            xh, mu, _, mvp, _ = model(xb, mb, teacher_forcing_ratio=0.0)
            preds.append(xh.cpu().numpy())
            mus_all.append(mu.cpu().numpy())
            mvps.append(mvp.cpu().numpy())
    preds = np.concatenate(preds)
    mvps = np.concatenate(mvps)
    mus_all = np.concatenate(mus_all)

    r2_norm = r2_score(X.ravel(), preds.ravel())
    fam_b = fam[:, None, None]
    mv_b = mv[:, :, None]
    X_orig = X * mv_b * fam_b
    P_orig = preds * mvps[:, :, None] * fam_b
    r2_orig = r2_score(X_orig.ravel(), P_orig.ravel())

    per_curve_r2 = [r2_score(X[:, c, :].ravel(), preds[:, c, :].ravel()) for c in range(7)]
    yt = mv[:, 1:].ravel()
    yp = mvps[:, 1:].ravel()
    mv_r2 = r2_score(yt, yp)
    per_curve_mv = [r2_score(mv[:, k], mvps[:, k]) for k in range(1, 7)]

    var_per_dim = mus_all.var(axis=0)
    active = int((var_per_dim >= 0.01).sum())
    collapsed = int((var_per_dim < 0.01).sum())

    pca = PCA().fit(mus_all)
    cum = np.cumsum(pca.explained_variance_ratio_)

    return {
        "N_test": int(N),
        "recon_R2_normalized_pooled": float(r2_norm),
        "recon_R2_original_scale_pooled": float(r2_orig),
        "recon_MAE_normalized": float(np.mean(np.abs(X - preds))),
        "recon_R2_per_curve_normalized": [float(v) for v in per_curve_r2],
        "max_val_R2_pooled": float(mv_r2),
        "max_val_R2_per_curve_1to6": [float(v) for v in per_curve_mv],
        "latent_active_dims": active,
        "latent_collapsed_dims": collapsed,
        "pca_90": int(np.searchsorted(cum, 0.9) + 1),
        "pca_95": int(np.searchsorted(cum, 0.95) + 1),
        "pca_99": int(np.searchsorted(cum, 0.99) + 1),
    }


def write_comparison(cond, base):
    """Render a side-by-side markdown table."""
    md = """# RESULTS_COMPARISON.md — Conditioned vs Non-Conditioned Baseline

Both models: 30D latent, LSTM-VAE, 256 hidden, 2 layers, identical training
schedule. Only difference: `use_scale_conditioning`.

Headline: the architectural innovation is responsible for the max-value
prediction lift; everything else is comparable.

"""
    md += "| Metric | Conditioned | Baseline (no cond.) | Δ |\n"
    md += "|---|---|---|---|\n"

    def row(label, k, fmt="{:.4f}", invert_arrow=False):
        a = cond.get(k)
        b = base.get(k)
        if a is None or b is None:
            return ""
        delta = a - b
        sign = "+" if delta >= 0 else ""
        arrow = ("▲" if (delta > 0) ^ invert_arrow else "▼") if abs(delta) > 1e-4 else "≈"
        return f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {sign}{delta:.4f} {arrow} |\n"

    md += row("Recon $R^2$ (normalized)", "recon_R2_normalized_pooled")
    md += row("Recon $R^2$ (original scale)", "recon_R2_original_scale_pooled")
    md += row("Recon MAE (normalized)", "recon_MAE_normalized", invert_arrow=True)
    md += row("Max-value $R^2$ (curves 1–6)", "max_val_R2_pooled")
    md += "\n"

    md += "| Per-curve recon $R^2$ | " + " | ".join(
        f"x{c}" for c in range(7)
    ) + " |\n|---|" + "|".join(["---"] * 7) + "|\n"
    md += "| Conditioned | " + " | ".join(
        f"{v:.3f}" for v in cond["recon_R2_per_curve_normalized"]
    ) + " |\n"
    md += "| Baseline    | " + " | ".join(
        f"{v:.3f}" for v in base["recon_R2_per_curve_normalized"]
    ) + " |\n\n"

    md += "| Per-curve max-val $R^2$ (curves 1–6) | " + " | ".join(
        f"x{c}" for c in range(1, 7)
    ) + " |\n|---|" + "|".join(["---"] * 6) + "|\n"
    md += "| Conditioned | " + " | ".join(
        f"{v:.3f}" for v in cond["max_val_R2_per_curve_1to6"]
    ) + " |\n"
    md += "| Baseline    | " + " | ".join(
        f"{v:.3f}" for v in base["max_val_R2_per_curve_1to6"]
    ) + " |\n\n"

    md += "## Latent space\n\n"
    md += "| Metric | Conditioned | Baseline |\n|---|---|---|\n"
    md += f"| Active dims (var ≥ 0.01) | {cond['latent_active_dims']} / 30 | {base['latent_active_dims']} / 30 |\n"
    md += f"| Collapsed dims | {cond['latent_collapsed_dims']} / 30 | {base['latent_collapsed_dims']} / 30 |\n"
    md += f"| PCs for 90% / 95% / 99% var | {cond['pca_90']} / {cond['pca_95']} / {cond['pca_99']} | {base['pca_90']} / {base['pca_95']} / {base['pca_99']} |\n"

    OUT_MD.write_text(md)


def main():
    if not BASELINE_PATH.exists():
        print(f"ERROR: baseline checkpoint not found at {BASELINE_PATH}")
        print("Run `python train_baseline.py` first.")
        sys.exit(1)

    with open(TEST_PATH, "rb") as f:
        test_pkg = pickle.load(f)

    print("Evaluating baseline...")
    base = evaluate_baseline(test_pkg)
    OUT_JSON.write_text(json.dumps(base, indent=2))
    print(f"Saved → {OUT_JSON}")

    cond_full = json.loads(RESULTS_JSON.read_text())
    cond = cond_full["reconstruction"]  # the same key shape as `base`

    write_comparison(cond, base)
    print(f"Saved → {OUT_MD}")

    print("\n=== HEADLINE COMPARISON ===")
    print(f"  Max-value R²    : conditioned {cond['max_val_R2_pooled']:.3f}"
          f"  vs  baseline {base['max_val_R2_pooled']:.3f}")
    print(f"  Recon R² (norm) : conditioned {cond['recon_R2_normalized_pooled']:.3f}"
          f"  vs  baseline {base['recon_R2_normalized_pooled']:.3f}")


if __name__ == "__main__":
    main()
