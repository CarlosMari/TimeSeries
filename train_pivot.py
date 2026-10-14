"""Wrapper around train_cvae.train() for the 2026-05-15 pivot.

Adds CLI arguments for:
  - --data-train, --data-test   (no-sort vs with-sort datasets)
  - --seed                       (so we can run 3 seeds per model)
  - --name                       (checkpoint name)
  - --epochs                     (default 1000; v1 used 2000, the baseline used 1000)
  - --use-scale-conditioning / --no-scale-conditioning  (models 1 vs 2)
  - --wandb-project              (default Conditional_LV_VAE_pivot)
  - --no-log                     (skip wandb)

Reuses train_cvae.train(). A scale contract must now be selected explicitly:
species_v2 is the corrected three-model path; legacy_sorted_v1 is historical.
No training is started by --help or importing this module.
"""

from __future__ import annotations

import argparse
import math
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
from src.models.cvae_heteroscedastic import HeteroscedasticLSTMVAE  # noqa: E402
from src.models.latent_ode import LatentODE  # noqa: E402
from src.models.transformer_vae import TransformerVAE  # noqa: E402
from src.models.kan_vae import KANVAE  # noqa: E402
from src.models.scale_contract import SCALE_CONTRACTS, SPECIES_SCALE_CONTRACT  # noqa: E402


MODEL_REGISTRY = {
    "cvae": LSTM_VAE,                              # models 1 + 2 (with / without scale-cond)
    "cvae-stochastic": StochasticLSTMVAE,          # model 3
    "latent-ode": LatentODE,                       # model 4
    "transformer-vae": TransformerVAE,             # model 5
    "kan-vae": KANVAE,                             # model 6
    # "glv-regression": GLVRegressor,              # model 7 — uses train_glv_regression.py
    "cvae-heteroscedastic": HeteroscedasticLSTMVAE,  # model 8 — per-(t,c) σ + Gaussian NLL
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
    ap.add_argument("--scale-contract", choices=SCALE_CONTRACTS, required=True,
                    help="species_v2 for corrected models; legacy_sorted_v1 for explicit reproduction")
    ap.add_argument("--scale-log-floor", type=float, default=1e-8,
                    help="recorded numerical floor for centered log-ratio targets (not a detection limit)")
    ap.add_argument("--allow-legacy-on-unsorted", action="store_true",
                    help="explicit historical reproduction only; retains the known six-output bug")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--use-scale-conditioning", dest="use_scale", action="store_true",
                    default=True)
    ap.add_argument("--no-scale-conditioning", dest="use_scale", action="store_false")
    ap.add_argument("--decoder-noise-init", type=float, default=0.05,
                    help="initial decoder-noise σ (only used for cvae-stochastic)")
    ap.add_argument("--decoder-noise-freeze", action="store_true",
                    help="freeze decoder-noise σ at the init value (don't learn). "
                         "Useful for the 'is decoder stochasticity actually testable' experiment.")
    ap.add_argument("--spectral-loss-weight", type=float, default=0.0,
                    help="add a spectral (FFT-magnitude MSE) term to the loss "
                         "with this weight. 0 = off (default, v1 behavior).")
    ap.add_argument("--wandb-project", default="Conditional_LV_VAE_pivot")
    ap.add_argument("--no-log", dest="log", action="store_false", default=True)
    args = ap.parse_args()

    if args.scale_contract == SPECIES_SCALE_CONTRACT and args.model not in {
        "cvae", "cvae-heteroscedastic", "transformer-vae",
    }:
        ap.error("species_v2 is implemented only for cvae, cvae-heteroscedastic, and transformer-vae")
    if args.allow_legacy_on_unsorted and args.scale_contract == SPECIES_SCALE_CONTRACT:
        ap.error("--allow-legacy-on-unsorted does not apply to species_v2")
    if not math.isfinite(args.scale_log_floor) or not 0 < args.scale_log_floor < 1:
        ap.error("--scale-log-floor must be finite and between 0 and 1")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        ap.error("--name must be a checkpoint basename, not a path")
    destination = Path(model_config["save_route"]) / f"{args.name}.pth"
    if destination.exists():
        ap.error(f"Refusing to overwrite existing checkpoint: {destination}")

    set_seed(args.seed)

    hp["random_seed"] = args.seed
    hp["epochs"] = args.epochs
    # β-warmup proportional to total epochs (was 300 of 2000 = 15%, keep proportion)
    hp["warmup_epochs"] = max(50, int(0.15 * args.epochs))

    model_config["use_scale_conditioning"] = args.use_scale
    model_config["name"] = args.name
    model_config["scale_contract"] = args.scale_contract
    model_config["scale_log_floor"] = args.scale_log_floor
    model_config["allow_legacy_on_unsorted"] = args.allow_legacy_on_unsorted
    if args.model == "cvae-stochastic":
        model_config["decoder_noise_init"] = args.decoder_noise_init
        model_config["decoder_noise_freeze"] = args.decoder_noise_freeze

    # Hook the optional spectral loss into train_cvae.vae_loss (default 0 = off)
    train_cvae.SPECTRAL_LOSS_WEIGHT = float(args.spectral_loss_weight)

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
    print(f"  scale:        {args.scale_contract}")
    print(f"  legacy NOSORT override: {args.allow_legacy_on_unsorted}")
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
