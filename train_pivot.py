"""Wrapper around train_cvae.train() for the 2026-05-15 pivot.

Adds CLI arguments for:
  - --data-train, --data-test   (no-sort vs with-sort datasets)
  - --seed                       (so we can run 3 seeds per model)
  - --name                       (checkpoint name)
  - --epochs                     (default 1000; v1 used 2000, the baseline used 1000)
  - --use-scale-conditioning / --no-scale-conditioning  (models 1 vs 2)
  - --wandb-project              (default Conditional_LV_VAE_pivot)
  - --no-log                     (skip wandb)

Reuses train_cvae.train() so v1 training (`python train_cvae.py`) stays
bit-identical and reproducible.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import train_cvae  # noqa: E402  — module-level imports include wandb, model, etc.
from src.utils.config import hp, model_config  # noqa: E402
from src.models.cvae import LSTM_VAE  # noqa: E402
from src.models.cvae_stochastic import StochasticLSTMVAE  # noqa: E402
from src.models.latent_ode import LatentODE  # noqa: E402


MODEL_REGISTRY = {
    "cvae": LSTM_VAE,                      # models 1 + 2 (with / without scale-cond)
    "cvae-stochastic": StochasticLSTMVAE,  # model 3
    "latent-ode": LatentODE,               # model 4
    # The following are added as their training scripts land:
    # "transformer-vae": TransformerVAE,   # model 5
    # "kan-vae": KANVAE,                   # model 6
    # "glv-regression": GLVRegressor,      # model 7 — uses its own trainer
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="cvae",
                    help="which architecture to train (default: cvae)")
    ap.add_argument("--data-train", default="data/TRAIN_FINAL_NOSORT.pkl")
    ap.add_argument("--data-test", default="data/TEST_FINAL_NOSORT.pkl")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", required=True, help="checkpoint name (no .pth)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--use-scale-conditioning", dest="use_scale", action="store_true",
                    default=True)
    ap.add_argument("--no-scale-conditioning", dest="use_scale", action="store_false")
    ap.add_argument("--decoder-noise-init", type=float, default=0.05,
                    help="initial decoder-noise σ (only used for cvae-stochastic)")
    ap.add_argument("--wandb-project", default="Conditional_LV_VAE_pivot")
    ap.add_argument("--no-log", dest="log", action="store_false", default=True)
    args = ap.parse_args()

    set_seed(args.seed)

    hp["random_seed"] = args.seed
    hp["epochs"] = args.epochs
    # β-warmup proportional to total epochs (was 300 of 2000 = 15%, keep proportion)
    hp["warmup_epochs"] = max(50, int(0.15 * args.epochs))

    model_config["use_scale_conditioning"] = args.use_scale
    model_config["name"] = args.name
    if args.model == "cvae-stochastic":
        model_config["decoder_noise_init"] = args.decoder_noise_init

    # Patch the data routes the train script uses (it reads module globals)
    train_cvae.TRAIN_ROUTE = args.data_train
    train_cvae.TEST_ROUTE = args.data_test
    train_cvae.LOG = args.log

    # Override the default wandb project. train_cvae.train() reads wandb.run is None
    # then calls wandb.init() — so we can pre-init here with our chosen project name
    # and let the train script see an already-initialized run.
    if args.log:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.name,
            config={**hp, **model_config, "_pivot": True, "seed": args.seed,
                    "data_train": args.data_train, "data_test": args.data_test},
            job_type="train",
        )

    print(f"=== Pivot training run ===")
    print(f"  model:        {args.model}")
    print(f"  seed:         {args.seed}")
    print(f"  name:         {args.name}")
    print(f"  epochs:       {args.epochs}")
    print(f"  scale-cond:   {args.use_scale}")
    print(f"  data-train:   {args.data_train}")
    print(f"  data-test:    {args.data_test}")
    print(f"  wandb proj:   {args.wandb_project if args.log else '(disabled)'}")

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(config=model_config)
    train_cvae.train(model)


if __name__ == "__main__":
    main()
