"""
Single source of truth for every number in the paper.

Loads the 30D scale-conditioned CVAE, evaluates the full test set, and writes
RESULTS.json + a Markdown table. All figure-generation scripts should read from
RESULTS.json rather than recomputing.

Usage:
    python analysis/produce_paper_metrics.py
    # writes RESULTS.json and RESULTS.md in the repo root
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.cvae import LSTM_VAE  # noqa: E402
from utils.post_processing import apply_extinction_threshold  # noqa: E402

MODEL_PATH = REPO_ROOT / "model_ckpts" / "model_final_30_conditioned.pth"
TEST_PATH = REPO_ROOT / "data" / "TEST_FINAL_PROCESSED.pkl"
TRAIN_PATH = REPO_ROOT / "data" / "TRAIN_FINAL_PROCESSED.pkl"

OUTPUT_JSON = REPO_ROOT / "RESULTS.json"
OUTPUT_MD = REPO_ROOT / "RESULTS.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.default_rng(0)

MODEL_CONFIG = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": "log",
    "use_scale_conditioning": True,
}

EXTINCTION_THRESHOLD = 0.005  # winner of the sweep (THRESHOLD_SWEEP_RESULTS.txt)


def load_model():
    model = LSTM_VAE(MODEL_CONFIG).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def bootstrap_ci(values: np.ndarray, fn=np.mean, n=1000, alpha=0.05):
    """Percentile bootstrap CI for a statistic `fn` applied to `values`."""
    n_obs = len(values)
    stats = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, n_obs, size=n_obs)
        stats[i] = fn(values[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate_reconstruction(model, test_pkg):
    X = test_pkg["data"]
    mv = test_pkg["reconstruction_max_values"]
    fam = test_pkg["family_max_values"]
    N = X.shape[0]

    preds, mvps, mus_all = [], [], []
    bs = 1000
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=DEVICE)
            mb = torch.tensor(mv[i : i + bs], dtype=torch.float32, device=DEVICE)
            xh, mu, _, mvp, _ = model(xb, mb, teacher_forcing_ratio=0.0)
            preds.append(xh.cpu().numpy())
            mus_all.append(mu.cpu().numpy())
            mvps.append(mvp.cpu().numpy())
    preds = np.concatenate(preds)
    mvps = np.concatenate(mvps)
    mus_all = np.concatenate(mus_all)

    # Per-sample R^2 on (7, 65) flattened, then aggregate
    flat_X = X.reshape(N, -1)
    flat_P = preds.reshape(N, -1)
    ss_res = ((flat_X - flat_P) ** 2).sum(axis=1)
    ss_tot = ((flat_X - flat_X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    per_sample_r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)

    r2_norm_pooled = r2_score(X.ravel(), preds.ravel())
    r2_norm_lo, r2_norm_hi = bootstrap_ci(per_sample_r2, np.mean)

    # Original scale
    fam_b = fam[:, None, None]
    mv_b = mv[:, :, None]
    X_orig = X * mv_b * fam_b
    P_orig = preds * mvps[:, :, None] * fam_b
    r2_orig_pooled = r2_score(X_orig.ravel(), P_orig.ravel())

    # Per curve
    per_curve_r2 = [r2_score(X[:, c, :].ravel(), preds[:, c, :].ravel()) for c in range(7)]

    # Max-value
    yt = mv[:, 1:].ravel()
    yp = mvps[:, 1:].ravel()
    mv_r2_pooled = r2_score(yt, yp)
    per_curve_mv = [r2_score(mv[:, k], mvps[:, k]) for k in range(1, 7)]

    # Bootstrap CI on max-val R^2
    n_samples = mv.shape[0]
    boot_mv = np.empty(1000)
    for i in range(1000):
        idx = RNG.integers(0, n_samples, size=n_samples)
        boot_mv[i] = r2_score(mv[idx, 1:].ravel(), mvps[idx, 1:].ravel())
    mv_r2_lo, mv_r2_hi = float(np.percentile(boot_mv, 2.5)), float(np.percentile(boot_mv, 97.5))

    # Latent health
    var_per_dim = mus_all.var(axis=0)
    active = int((var_per_dim >= 0.01).sum())
    collapsed = int((var_per_dim < 0.01).sum())

    pca = PCA().fit(mus_all)
    cum = np.cumsum(pca.explained_variance_ratio_)
    pcs = {
        "pca_90": int(np.searchsorted(cum, 0.9) + 1),
        "pca_95": int(np.searchsorted(cum, 0.95) + 1),
        "pca_99": int(np.searchsorted(cum, 0.99) + 1),
    }

    return {
        "N_test": int(N),
        "recon_R2_normalized_pooled": float(r2_norm_pooled),
        "recon_R2_normalized_per_sample_mean": float(per_sample_r2.mean()),
        "recon_R2_normalized_95CI": [r2_norm_lo, r2_norm_hi],
        "recon_R2_original_scale_pooled": float(r2_orig_pooled),
        "recon_MAE_normalized": float(np.mean(np.abs(X - preds))),
        "recon_MSE_normalized": float(np.mean((X - preds) ** 2)),
        "recon_R2_per_curve_normalized": [float(v) for v in per_curve_r2],
        "max_val_R2_pooled": float(mv_r2_pooled),
        "max_val_R2_95CI": [mv_r2_lo, mv_r2_hi],
        "max_val_R2_per_curve_1to6": [float(v) for v in per_curve_mv],
        "latent_var_per_dim": [float(v) for v in var_per_dim],
        "latent_active_dims": active,
        "latent_collapsed_dims": collapsed,
        **pcs,
    }, mus_all, preds, mvps


def lv_adherence_per_sample(traj_t_s: np.ndarray) -> float:
    """LV adherence test for ONE trajectory of shape (T, S).

    Returns the mean across-species R^2 of the linear regression
    d(log x_i)/dt ~ b0 + sum_j a_ij x_j.
    """
    T, S = traj_t_s.shape
    x = np.clip(traj_t_s, 1e-9, None)
    log_x = np.log(x)
    dlog = np.gradient(log_x, axis=0)  # (T, S)
    r2s = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(S):
            reg = LinearRegression().fit(x, dlog[:, s])
            pred = reg.predict(x)
            r2s.append(r2_score(dlog[:, s], pred))
    return float(np.mean(r2s))


def evaluate_lv_adherence(model, test_pkg, n_eval=2000):
    """Real test data vs generated samples on LV adherence."""
    X = test_pkg["data"]
    mv = test_pkg["reconstruction_max_values"]
    fam = test_pkg["family_max_values"]

    idx = RNG.choice(X.shape[0], size=n_eval, replace=False)
    real_orig = (X[idx] * mv[idx][:, :, None]) * fam[idx][:, None, None]
    real_ts = np.transpose(real_orig, (0, 2, 1))  # (n, T, S)
    r2_real = np.array([lv_adherence_per_sample(real_ts[i]) for i in range(n_eval)])

    # Generated
    torch.manual_seed(7)
    with torch.no_grad():
        zs = torch.randn(n_eval, MODEL_CONFIG["latent_dim"], device=DEVICE)
        gen_norm, gen_mv = model.decode(zs)
    gen_norm = gen_norm.cpu().numpy()
    gen_mv = gen_mv.cpu().numpy()
    fam_sampled = RNG.choice(fam, size=n_eval, replace=True)
    gen_orig = gen_norm * gen_mv[:, :, None] * fam_sampled[:, None, None]
    gen_ts = np.transpose(gen_orig, (0, 2, 1))

    r2_gen_raw = np.array([lv_adherence_per_sample(gen_ts[i]) for i in range(n_eval)])
    gen_ts_fix = np.array(
        [apply_extinction_threshold(gen_ts[i], threshold=EXTINCTION_THRESHOLD) for i in range(n_eval)]
    )
    r2_gen_fix = np.array([lv_adherence_per_sample(gen_ts_fix[i]) for i in range(n_eval)])

    def summarize(arr):
        m = float(arr.mean())
        ci_lo, ci_hi = bootstrap_ci(arr, np.mean)
        return {
            "mean": m,
            "mean_95CI": [ci_lo, ci_hi],
            "median": float(np.median(arr)),
            "pct_gt_09": float((arr > 0.9).mean()),
            "pct_gt_095": float((arr > 0.95).mean()),
        }

    return {
        "n_eval": n_eval,
        "extinction_threshold": EXTINCTION_THRESHOLD,
        "real": summarize(r2_real),
        "generated_raw": summarize(r2_gen_raw),
        "generated_with_fix": summarize(r2_gen_fix),
    }


def evaluate_novelty(model, n_gen=1000, n_ref=20000):
    """Distance of generated samples to nearest train / test samples."""
    with open(TRAIN_PATH, "rb") as f:
        train_full = pickle.load(f)["data"].reshape(-1, 7 * 65)
    with open(TEST_PATH, "rb") as f:
        test_full = pickle.load(f)["data"].reshape(-1, 7 * 65)

    train_sub = train_full[RNG.choice(len(train_full), size=n_ref, replace=False)]
    test_sub = test_full[RNG.choice(len(test_full), size=n_ref, replace=False)]

    torch.manual_seed(42)
    with torch.no_grad():
        z = torch.randn(n_gen, MODEL_CONFIG["latent_dim"], device=DEVICE)
        gen, _ = model.decode(z)
    gen = gen.cpu().numpy().reshape(n_gen, -1)

    nn_train = NearestNeighbors(n_neighbors=1).fit(train_sub)
    nn_test = NearestNeighbors(n_neighbors=1).fit(test_sub)
    d_gen_train, _ = nn_train.kneighbors(gen)
    d_gen_test, _ = nn_test.kneighbors(gen)
    d_te_train, _ = nn_train.kneighbors(test_sub[:n_gen])

    nn_gen = NearestNeighbors(n_neighbors=2).fit(gen)
    d_internal = nn_gen.kneighbors(gen)[0][:, 1]

    return {
        "n_gen": n_gen,
        "n_ref": n_ref,
        "gen_to_train_mean": float(d_gen_train.mean()),
        "gen_to_test_mean": float(d_gen_test.mean()),
        "test_to_train_mean": float(d_te_train.mean()),
        "gen_internal_mean": float(d_internal.mean()),
        "memorization_ratio": float(d_gen_train.mean() / d_te_train.mean()),
    }


def write_markdown(results: dict):
    recon = results["reconstruction"]
    lv = results["lv_adherence"]
    nov = results["novelty"]
    md = f"""# RESULTS.md — Verified Paper Metrics

Generated by `analysis/produce_paper_metrics.py`. **All numbers in the paper should come from here.**

Checkpoint: `model_ckpts/model_final_30_conditioned.pth`
Test set: N = {recon['N_test']} samples
Extinction threshold (for generated): θ = {lv['extinction_threshold']}

## Reconstruction

| Metric | Value | 95% CI |
|---|---|---|
| $R^2$ normalized (pooled, all timepoints) | {recon['recon_R2_normalized_pooled']:.4f} | — |
| $R^2$ normalized (per-sample mean) | {recon['recon_R2_normalized_per_sample_mean']:.4f} | [{recon['recon_R2_normalized_95CI'][0]:.4f}, {recon['recon_R2_normalized_95CI'][1]:.4f}] |
| $R^2$ original scale (pooled) | {recon['recon_R2_original_scale_pooled']:.4f} | — |
| MAE normalized | {recon['recon_MAE_normalized']:.4f} | — |
| MSE normalized | {recon['recon_MSE_normalized']:.5f} | — |

Per-curve $R^2$ (normalized): {", ".join(f"{v:.3f}" for v in recon['recon_R2_per_curve_normalized'])}

## Max-value prediction (curves 1–6)

| Metric | Value | 95% CI |
|---|---|---|
| Pooled $R^2$ | {recon['max_val_R2_pooled']:.4f} | [{recon['max_val_R2_95CI'][0]:.4f}, {recon['max_val_R2_95CI'][1]:.4f}] |

Per-curve $R^2$: {", ".join(f"{v:.3f}" for v in recon['max_val_R2_per_curve_1to6'])}

## Latent space

| Metric | Value |
|---|---|
| Active dims (var ≥ 0.01) | {recon['latent_active_dims']} / 30 |
| Collapsed dims | {recon['latent_collapsed_dims']} / 30 |
| PCs for 90% / 95% / 99% var | {recon['pca_90']} / {recon['pca_95']} / {recon['pca_99']} |

## LV adherence (n = {lv['n_eval']})

| Source | Mean | 95% CI | Median | % > 0.9 | % > 0.95 |
|---|---|---|---|---|---|
| Real test data | {lv['real']['mean']:.4f} | [{lv['real']['mean_95CI'][0]:.4f}, {lv['real']['mean_95CI'][1]:.4f}] | {lv['real']['median']:.4f} | {100*lv['real']['pct_gt_09']:.1f}% | {100*lv['real']['pct_gt_095']:.1f}% |
| Generated (raw) | {lv['generated_raw']['mean']:.4f} | [{lv['generated_raw']['mean_95CI'][0]:.4f}, {lv['generated_raw']['mean_95CI'][1]:.4f}] | {lv['generated_raw']['median']:.4f} | {100*lv['generated_raw']['pct_gt_09']:.1f}% | {100*lv['generated_raw']['pct_gt_095']:.1f}% |
| Generated (θ={lv['extinction_threshold']}) | {lv['generated_with_fix']['mean']:.4f} | [{lv['generated_with_fix']['mean_95CI'][0]:.4f}, {lv['generated_with_fix']['mean_95CI'][1]:.4f}] | {lv['generated_with_fix']['median']:.4f} | {100*lv['generated_with_fix']['pct_gt_09']:.1f}% | {100*lv['generated_with_fix']['pct_gt_095']:.1f}% |

## Novelty (nearest-neighbor distance, Euclidean on flattened curves)

| Distance | Mean |
|---|---|
| Generated → nearest train | {nov['gen_to_train_mean']:.4f} |
| Generated → nearest test | {nov['gen_to_test_mean']:.4f} |
| Test → nearest train (baseline) | {nov['test_to_train_mean']:.4f} |
| Generated internal NN | {nov['gen_internal_mean']:.4f} |
| Memorization ratio (gen/train ÷ test/train) | **{nov['memorization_ratio']:.3f}** |
"""
    OUTPUT_MD.write_text(md)


def main():
    print("Loading model...")
    model = load_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    print("Loading test data...")
    with open(TEST_PATH, "rb") as f:
        test_pkg = pickle.load(f)

    print("Evaluating reconstruction + latent...")
    recon_results, mus_all, preds, mvps = evaluate_reconstruction(model, test_pkg)

    print("Evaluating LV adherence...")
    lv_results = evaluate_lv_adherence(model, test_pkg, n_eval=2000)

    print("Evaluating novelty...")
    nov_results = evaluate_novelty(model)

    results = {
        "checkpoint": str(MODEL_PATH.relative_to(REPO_ROOT)),
        "model_config": MODEL_CONFIG,
        "reconstruction": recon_results,
        "lv_adherence": lv_results,
        "novelty": nov_results,
    }

    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    write_markdown(results)
    print(f"Wrote {OUTPUT_JSON.relative_to(REPO_ROOT)} and {OUTPUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
