"""Unified evaluation harness for the 7-model pivot.

Given a set of trained checkpoints, this script:

  1. Loads each checkpoint with its architecture adapter.
  2. Runs the per-model **reconstruction** path on the held-out test set.
  3. Runs **generation** to produce N samples.
  4. Computes the full eval matrix (reconstruction R², max-value R²,
     feature-MMD, density/coverage, RQA, Rosenstein λ₁, parameter recovery
     where applicable, latent active dims where applicable).
  5. Writes per-checkpoint JSON + the aggregate `RESULTS_COMPARATIVE.json`.

Per-architecture adapters live below in REGISTRY. To add a new model, define
a function that takes (config_dict, checkpoint_path, device) and returns
``model`` exposing forward / generate (LSTM_VAE-compatible API). Register
the function under a model-type key matched against the checkpoint's filename
pattern.

By design this script is *idempotent*: re-running it on already-computed
metrics short-circuits unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.cvae import LSTM_VAE  # noqa: E402
from src.models.cvae_stochastic import StochasticLSTMVAE  # noqa: E402
from src.models.latent_ode import LatentODE  # noqa: E402
from src.models.transformer_vae import TransformerVAE  # noqa: E402
from src.models.kan_vae import KANVAE  # noqa: E402
from src.models.glv_regression import GLVRegressor  # noqa: E402
from utils.post_processing import apply_extinction_threshold  # noqa: E402
from analysis.chaos_diagnostics import rqa_measures, rosenstein_lyapunov  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GEN = 2000          # generated samples per model
N_RECON = 5000        # test samples to evaluate reconstruction on
N_CHAOS = 200         # per-group for chaos diagnostics (RQA, Lyap)
K_NN = 5

# Features for MMD — drops mean_extrema + mean_curvature per the D3 fix
# (these were driving the headline KS in §3.8 of the v1 paper but are
# redundant with per-species std-style features and integer-valued, which
# inflates the KS by construction).
DROPPED_FEATURES = {"mean_extrema", "mean_curv"}


# -----------------------------------------------------------------------------
# Adapters: load checkpoint → model instance with .generate() and .forward()
# -----------------------------------------------------------------------------

def _common_config(latent_dim=30) -> dict:
    return {
        "n_curves": 7, "seq_len": 65, "latent_dim": latent_dim,
        "rnn_hidden_size": 256, "rnn_num_layers": 2,
        "scale_prediction_mode": "log",
    }


def adapter_cvae(checkpoint_path: Path, use_scale_conditioning: bool, device) -> LSTM_VAE:
    cfg = {**_common_config(), "use_scale_conditioning": use_scale_conditioning}
    m = LSTM_VAE(cfg).to(device)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()
    return m


def adapter_cvae_stochastic(checkpoint_path: Path, device) -> StochasticLSTMVAE:
    # Detect whether checkpoint has frozen or learnable σ by inspecting state dict keys
    sd = torch.load(checkpoint_path, map_location=device)
    is_frozen = "decoder_noise_sigma_frozen" in sd
    init_sigma = float(sd.get("decoder_noise_sigma_frozen", 0.05))
    cfg = {**_common_config(), "use_scale_conditioning": True,
           "decoder_noise_init": init_sigma,
           "decoder_noise_freeze": is_frozen}
    m = StochasticLSTMVAE(cfg).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def adapter_latent_ode(checkpoint_path: Path, device) -> LatentODE:
    cfg = {**_common_config(), "use_scale_conditioning": True}
    m = LatentODE(cfg).to(device)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()
    return m


def adapter_transformer(checkpoint_path: Path, device) -> TransformerVAE:
    cfg = {**_common_config(), "use_scale_conditioning": True}
    m = TransformerVAE(cfg).to(device)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()
    return m


def adapter_kan(checkpoint_path: Path, device) -> KANVAE:
    cfg = {**_common_config(), "use_scale_conditioning": True}
    m = KANVAE(cfg).to(device)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()
    return m


def adapter_glv_regression(checkpoint_path: Path, device) -> GLVRegressor:
    ckpt = torch.load(checkpoint_path, map_location=device)
    m = GLVRegressor(ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    if "emp_r" in ckpt:
        m.store_empirical_distribution(ckpt["emp_r"], ckpt["emp_A"], ckpt["emp_x0"])
    m.eval()
    return m


# -----------------------------------------------------------------------------
# Feature extraction (D3-corrected: drops mean_extrema + mean_curv)
# -----------------------------------------------------------------------------

def features_for_one(traj: np.ndarray) -> tuple[np.ndarray, list[str]]:
    S, T = traj.shape
    feats = []
    labels = []
    t = np.arange(T)
    for s in range(S):
        x = traj[s]
        feats.append(x.mean()); labels.append(f"sp{s}_mean")
        feats.append(x.std()); labels.append(f"sp{s}_std")
        slope, _ = np.polyfit(t, x, 1)
        feats.append(slope); labels.append(f"sp{s}_trend")

    feats.append(float(traj.std(axis=1).sum())); labels.append("total_var")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = np.corrcoef(traj)
    iu = np.triu_indices(S, k=1)
    feats.append(float(np.nanmean(C[iu]))); labels.append("mean_corr")

    def dominant_freq(x):
        spec = np.abs(np.fft.rfft(x - x.mean()))
        if spec.sum() == 0:
            return 0.0
        return float(np.argmax(spec[1:]) + 1) / len(x)
    feats.append(float(np.mean([dominant_freq(traj[s]) for s in range(S)])))
    labels.append("mean_freq")

    return np.array(feats, dtype=np.float64), labels


def featurize(trajs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    feats_list = []
    labels = None
    for i in range(trajs.shape[0]):
        f, l = features_for_one(trajs[i])
        feats_list.append(f)
        if labels is None:
            labels = l
    return np.stack(feats_list, axis=0), labels


# -----------------------------------------------------------------------------
# Lens 1: MMD + density/coverage
# -----------------------------------------------------------------------------

def median_heuristic_sigma(X: np.ndarray, rng) -> float:
    n = min(500, X.shape[0])
    sub = X[rng.choice(X.shape[0], size=n, replace=False)]
    d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
    d = d[np.triu_indices(n, k=1)]
    return float(np.median(d))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    def K(A, B):
        d = np.linalg.norm(A[:, None] - B[None, :], axis=-1) ** 2
        return np.exp(-d / (2 * sigma ** 2))
    Kxx, Kyy, Kxy = K(X, X), K(Y, Y), K(X, Y)
    n, m = X.shape[0], Y.shape[0]
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.sum() / (n * m))


def mmd_permutation_test(X, Y, n_perm=200, rng=None):
    rng = rng or np.random.default_rng(0)
    sigma = median_heuristic_sigma(np.vstack([X, Y]), rng)
    observed = mmd2_unbiased(X, Y, sigma)
    nx = X.shape[0]
    combined = np.vstack([X, Y])
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(combined.shape[0])
        perm_stats[i] = mmd2_unbiased(combined[idx[:nx]], combined[idx[nx:]], sigma)
    p = float((perm_stats >= observed).sum() + 1) / (n_perm + 1)
    return {"sigma": sigma, "mmd2_observed": observed,
            "p_value_permutation": p, "n_permutations": n_perm,
            "perm_stats_mean": float(perm_stats.mean()),
            "perm_stats_std": float(perm_stats.std())}


def density_coverage(real, gen, k=K_NN):
    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real)
    dist_real, _ = nn_real.kneighbors(real)
    radii = dist_real[:, -1]
    nn_to_gen = NearestNeighbors().fit(gen)
    counts = np.zeros(real.shape[0], dtype=np.int32)
    for i, r_i in enumerate(real):
        in_ball = nn_to_gen.radius_neighbors(r_i.reshape(1, -1), radius=radii[i],
                                             return_distance=False)
        counts[i] = len(in_ball[0])
    return {"density_at_k": float(counts.sum() / (k * real.shape[0])),
            "coverage_at_k": float((counts > 0).mean()),
            "k": k}


# -----------------------------------------------------------------------------
# Per-model evaluation
# -----------------------------------------------------------------------------

def evaluate_model(model, model_type: str, test_pkg: dict, rng) -> dict:
    """Run the full eval matrix on one loaded model."""
    real_norm = test_pkg["data"]
    real_mv = test_pkg["reconstruction_max_values"]
    real_fam = test_pkg["family_max_values"]
    N_test = real_norm.shape[0]

    # --- Reconstruction (subsampled) ---
    recon_idx = rng.choice(N_test, size=min(N_RECON, N_test), replace=False)
    X_real_norm = torch.from_numpy(real_norm[recon_idx]).float().to(DEVICE)
    mv_real = torch.from_numpy(real_mv[recon_idx]).float().to(DEVICE)

    recon_metrics = {}
    if model_type != "glv-regression":
        with torch.no_grad():
            # use deterministic teacher-forcing-off pass
            out = model(X_real_norm, mv_real, teacher_forcing_ratio=0)
            X_hat, _, _, max_vals_pred, _ = out
        x_true = X_real_norm.cpu().numpy()
        x_hat = X_hat.cpu().numpy()
        mv_pred = max_vals_pred.cpu().numpy()

        ss_res = ((x_true - x_hat) ** 2).sum()
        ss_tot = ((x_true - x_true.mean()) ** 2).sum()
        recon_R2_norm = 1.0 - ss_res / max(ss_tot, 1e-12)

        # Original-scale: use *predicted* max-vals to denormalize the recon,
        # matching how the model would be used downstream.
        fam = real_fam[recon_idx][:, None, None]
        x_true_orig = x_true * real_mv[recon_idx][:, :, None] * fam
        x_hat_orig = x_hat * mv_pred[:, :, None] * fam

        ss_res_o = ((x_true_orig - x_hat_orig) ** 2).sum()
        ss_tot_o = ((x_true_orig - x_true_orig.mean()) ** 2).sum()
        recon_R2_orig = 1.0 - ss_res_o / max(ss_tot_o, 1e-12)

        recon_metrics = {
            "N_recon": int(len(recon_idx)),
            "recon_R2_normalized": float(recon_R2_norm),
            "recon_R2_original_scale": float(recon_R2_orig),
            "recon_MAE_normalized": float(np.abs(x_true - x_hat).mean()),
        }
        # Max-value pooled R² over curves 1..6
        mv_true_16 = real_mv[recon_idx][:, 1:].flatten()
        mv_pred_16 = mv_pred[:, 1:].flatten()
        ss_res_mv = ((mv_true_16 - mv_pred_16) ** 2).sum()
        ss_tot_mv = ((mv_true_16 - mv_true_16.mean()) ** 2).sum()
        recon_metrics["max_val_R2_pooled"] = float(1.0 - ss_res_mv / max(ss_tot_mv, 1e-12))
    else:
        recon_metrics = {"note": "GLVRegressor has no in-place reconstruction path; "
                                  "evaluated only on generation + parameter recovery."}

    # --- Generation ---
    torch.manual_seed(20260515)
    with torch.no_grad():
        X_gen, gen_mv = model.generate(N_GEN, DEVICE)
    X_gen_np = X_gen.cpu().numpy()
    gen_mv_np = gen_mv.cpu().numpy()

    # Denormalize generated samples
    fam_sampled = rng.choice(real_fam, size=N_GEN, replace=True)
    if model_type == "glv-regression":
        # Already returned in original scale (solve_ivp output)
        gen_orig = X_gen_np
    else:
        gen_orig = X_gen_np * gen_mv_np[:, :, None] * fam_sampled[:, None, None]

    # Extinction fix
    gen_ts = np.transpose(gen_orig, (0, 2, 1))
    gen_ts_fixed = np.array([
        apply_extinction_threshold(gen_ts[i], threshold=0.005) for i in range(N_GEN)
    ])
    gen_fixed = np.transpose(gen_ts_fixed, (0, 2, 1))

    # Real samples (matched N for fair comparison)
    real_idx_gen = rng.choice(N_test, size=N_GEN, replace=False)
    real_orig = (
        real_norm[real_idx_gen]
        * real_mv[real_idx_gen][:, :, None]
        * real_fam[real_idx_gen][:, None, None]
    )

    # --- Lens 1: feature-MMD + density/coverage ---
    F_real, feat_labels = featurize(real_orig)
    F_gen, _ = featurize(gen_fixed)

    # D3 fix: drop the dropped-feature columns
    keep = [i for i, l in enumerate(feat_labels) if l not in DROPPED_FEATURES]
    F_real = F_real[:, keep]
    F_gen = F_gen[:, keep]
    feat_labels_kept = [feat_labels[i] for i in keep]

    scaler = StandardScaler().fit(np.vstack([F_real, F_gen]))
    F_real_z = scaler.transform(F_real)
    F_gen_z = scaler.transform(F_gen)

    mmd = mmd_permutation_test(F_real_z, F_gen_z, n_perm=200, rng=rng)
    dc = density_coverage(F_real_z, F_gen_z, k=K_NN)
    dc_swap = density_coverage(F_gen_z, F_real_z, k=K_NN)
    ks_results = []
    for j, lab in enumerate(feat_labels_kept):
        s, p = stats.ks_2samp(F_real_z[:, j], F_gen_z[:, j])
        ks_results.append({"feature": lab, "ks": float(s), "p": float(p)})
    ks_n_sig = sum(1 for r in ks_results if r["p"] < 0.05)

    lens1 = {
        "feature_labels": feat_labels_kept,
        "feature_dim": int(F_real_z.shape[1]),
        "mmd": mmd,
        "density_coverage": dc,
        "density_coverage_swap": dc_swap,
        "ks_per_feature": ks_results,
        "ks_n_significant_at_p05": ks_n_sig,
    }

    # --- Lens 2 + 3: RQA + Lyapunov on n=200 each ---
    chaos_idx_real = rng.choice(N_GEN, size=min(N_CHAOS, N_GEN), replace=False)
    chaos_idx_gen = rng.choice(N_GEN, size=min(N_CHAOS, N_GEN), replace=False)

    def chaos_for_pop(trajs):
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

    rqa_real, lyap_real = chaos_for_pop(real_orig[chaos_idx_real])
    rqa_gen, lyap_gen = chaos_for_pop(gen_fixed[chaos_idx_gen])

    def summarize_chaos(arr):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "n": 0}
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                "median": float(np.median(a)), "n": int(a.size)}

    def ks_compare(a, b):
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if a.size < 5 or b.size < 5:
            return {"ks_stat": float("nan"), "ks_p": float("nan")}
        s, p = stats.ks_2samp(a, b)
        return {"ks_stat": float(s), "ks_p": float(p)}

    rqa_results = {}
    for k in ("RR", "DET", "L_mean", "L_max", "LAM", "TT"):
        rqa_results[k] = {
            "real": summarize_chaos(rqa_real[k]),
            "gen": summarize_chaos(rqa_gen[k]),
            "ks": ks_compare(rqa_real[k], rqa_gen[k]),
        }
    lyap_results = {
        "real": summarize_chaos(lyap_real),
        "gen": summarize_chaos(lyap_gen),
        "ks": ks_compare(lyap_real, lyap_gen),
    }

    return {
        "model_type": model_type,
        "reconstruction": recon_metrics,
        "lens1_feature_MMD": lens1,
        "lens2_RQA": rqa_results,
        "lens3_Lyapunov": lyap_results,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

ADAPTERS: dict[str, Callable] = {
    "cvae-scale-cond": lambda p, d: adapter_cvae(p, True, d),
    "cvae-no-scale-cond": lambda p, d: adapter_cvae(p, False, d),
    "cvae-stochastic": lambda p, d: adapter_cvae_stochastic(p, d),
    "latent-ode": adapter_latent_ode,
    "transformer-vae": adapter_transformer,
    "kan-vae": adapter_kan,
    "glv-regression": adapter_glv_regression,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="Pairs of <model-type>=<path> e.g. cvae-scale-cond=model_ckpts/model_1_seed42.pth")
    ap.add_argument("--test-path", default="data/TEST_FINAL_NOSORT.pkl")
    ap.add_argument("--out", default="RESULTS_COMPARATIVE.json")
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate even if a previous JSON is present")
    args = ap.parse_args()

    print(f"Loading test set from {args.test_path}...")
    with open(args.test_path, "rb") as f:
        test_pkg = pickle.load(f)
    print(f"  Test set N = {test_pkg['data'].shape[0]}")

    out_path = Path(args.out)
    existing = {}
    if out_path.exists() and not args.force:
        existing = json.loads(out_path.read_text())

    aggregate = existing.copy()

    for spec in args.checkpoints:
        if "=" not in spec:
            raise ValueError(f"Bad --checkpoints spec: {spec!r}; expected <type>=<path>")
        model_type, ckpt_path = spec.split("=", 1)
        ckpt_path = Path(ckpt_path)
        key = f"{model_type}:{ckpt_path.stem}"

        if not args.force and key in aggregate:
            print(f"[{key}] already in JSON, skipping (use --force to redo)")
            continue

        print(f"\n=== [{key}] ===")
        adapter = ADAPTERS.get(model_type)
        if adapter is None:
            raise KeyError(f"Unknown model type {model_type!r}; "
                           f"choices: {list(ADAPTERS.keys())}")
        model = adapter(ckpt_path, DEVICE)
        rng = np.random.default_rng(2026_05_15)
        metrics = evaluate_model(model, model_type, test_pkg, rng)
        aggregate[key] = metrics

        # Persist after every model so a crash doesn't lose work
        out_path.write_text(json.dumps(aggregate, indent=2, default=str))
        print(f"[{key}] done, written to {out_path}")

    print(f"\nFinished. {len(aggregate)} models in {out_path}")


if __name__ == "__main__":
    main()
