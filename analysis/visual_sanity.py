"""Quick visual sanity check: plot real + generated trajectories from each model.

Per the pre-multi-seed sanity-check plan (A2, A3): I haven't eyeballed a single
generated trajectory from any of the 7 seed-42 models. The numbers say all 7
produce "smoother-than-real" samples — let's see what that looks like.

Also doubles as A3: confirms each checkpoint produces non-garbage outputs.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.evaluate_all_models import ADAPTERS  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 4

MODELS = [
    ("cvae-scale-cond", "model_ckpts/model_1_seed42.pth", "m1 scale-cond VAE"),
    ("cvae-no-scale-cond", "model_ckpts/model_2_seed42.pth", "m2 no-cond VAE"),
    ("cvae-stochastic", "model_ckpts/model_3_seed42.pth", "m3 stochastic VAE"),
    ("latent-ode", "model_ckpts/model_4_seed42.pth", "m4 Latent-ODE"),
    ("transformer-vae", "model_ckpts/model_5_seed42.pth", "m5 Transformer-VAE"),
    ("kan-vae", "model_ckpts/model_6_seed42.pth", "m6 KAN-VAE"),
    ("glv-regression", "model_ckpts/model_7_seed42.pth", "m7 GLV-regression"),
]


def main():
    # Real samples (from the eval harness's RNG choice so they're representative)
    with open("data/TEST_FINAL_NOSORT.pkl", "rb") as f:
        pkg = pickle.load(f)
    rng = np.random.default_rng(2026_05_15)
    real_idx = rng.choice(pkg["data"].shape[0], size=N_SAMPLES, replace=False)
    real_orig = (
        pkg["data"][real_idx]
        * pkg["reconstruction_max_values"][real_idx][:, :, None]
        * pkg["family_max_values"][real_idx][:, None, None]
    )

    fig, axes = plt.subplots(len(MODELS) + 1, N_SAMPLES, figsize=(N_SAMPLES * 3.0, (len(MODELS) + 1) * 2.0),
                              sharex=True)

    # Top row: real
    for j in range(N_SAMPLES):
        ax = axes[0, j]
        for s in range(7):
            ax.plot(real_orig[j, s], lw=1.0)
        ax.set_title(f"real #{j}", fontsize=8)
        ax.set_ylabel("real", fontsize=9, rotation=0, ha="right", va="center") if j == 0 else None
        ax.tick_params(labelsize=7)

    # One row per model
    for i, (mtype, ckpt, lbl) in enumerate(MODELS, start=1):
        try:
            model = ADAPTERS[mtype](Path(ckpt), DEVICE)
            torch.manual_seed(42 + i)
            with torch.no_grad():
                X_gen, _ = model.generate(N_SAMPLES, DEVICE)
            X_gen = X_gen.cpu().numpy()

            # For VAE-family models: denormalize using a sampled family-max
            # (GLV-regression already returns original scale)
            if mtype != "glv-regression":
                fam = rng.choice(pkg["family_max_values"], size=N_SAMPLES, replace=True)
                # Use generated max_vals from the second return — but we already
                # need to re-call generate to get max_vals. Easier: just plot in
                # normalized space for VAE-family, and the actually-generated
                # values for GLV-regression.
                pass

            for j in range(N_SAMPLES):
                ax = axes[i, j]
                for s in range(7):
                    ax.plot(X_gen[j, s], lw=1.0)
                ax.set_title(f"{lbl} #{j}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(lbl.replace(" ", "\n"), fontsize=8, rotation=0, ha="right", va="center")
                ax.tick_params(labelsize=7)
                ax.set_ylim(bottom=0)
        except Exception as e:
            for j in range(N_SAMPLES):
                ax = axes[i, j]
                ax.text(0.5, 0.5, f"FAIL\n{type(e).__name__}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="red")
                ax.axis("off")
            print(f"WARN: {lbl} failed: {e}")

    fig.suptitle("Visual sanity: real (top row) + 1 generation per architecture (subsequent rows)",
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    out = REPO_ROOT / "final figures" / "fig_visual_sanity_seed42.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
