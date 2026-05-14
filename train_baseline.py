"""
Train the non-conditioned baseline VAE for the paper.

Same architecture, data, schedule as the conditioned model, but with
`use_scale_conditioning=False`. This is the reviewer-mandated baseline:
isolates the effect of the scale-conditioning architectural change.

Outputs: model_ckpts/model_final_30_baseline.pth

Reproducible from scratch with `python train_baseline.py`.
"""

import torch  # noqa: F401
import os
import wandb

# Mutate config BEFORE importing the trainer (trainer imports model_config at module load)
from src.utils.config import hp, model_config  # noqa: E402

model_config["use_scale_conditioning"] = False
model_config["name"] = "model_final_30_baseline"

# Faster training schedule for the baseline. 1000 epochs is enough to finish
# both TF decay (0.4 * epochs = 400 epoch window) and β warmup (300 → 1000
# is plenty of time at the plateau). The original 2000-epoch run mostly
# fine-tuned after epoch 800; we can verify by evaluating at convergence
# instead of waiting for an arbitrary cutoff.
hp["epochs"] = 1000

# Disable wandb auto-init for an offline run if needed; keep it on by default
# so we can see training curves. If WANDB_API_KEY is not set, force offline.
if "WANDB_API_KEY" not in os.environ and "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"

from src.models.cvae import LSTM_VAE  # noqa: E402
from train_cvae import train  # noqa: E402

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING NON-CONDITIONED BASELINE")
    print(f"  name              : {model_config['name']}")
    print(f"  use_scale_cond    : {model_config['use_scale_conditioning']}")
    print(f"  latent_dim        : {model_config['latent_dim']}")
    print(f"  epochs            : {hp['epochs']}")
    print("=" * 60)

    if "fingerprint_dim" in model_config:
        del model_config["fingerprint_dim"]

    # Override wandb job name so it shows up distinctly
    if wandb.run is None:
        wandb.init(
            project="Conditional_LV_VAE",
            config={**hp, **model_config},
            job_type="train_baseline",
            name="baseline_30D_non_conditioned",
        )

    vae_model = LSTM_VAE(config=model_config)
    train(vae_model)
