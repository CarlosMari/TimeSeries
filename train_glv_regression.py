"""Trainer for the GLVRegressor model (model 7).

Loss: MSE on (r, A) ground-truth from the matched dataset. Optional auxiliary
loss: integrate (r̂, Â) and MSE against the ground-truth trajectory (closes
the loop end-to-end). The auxiliary term is expensive (CPU solve_ivp) so it's
optional and weighted by --aux-weight.

Usage:
    python train_glv_regression.py --seed 42 --name model_7_seed42 --epochs 200
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.glv_regression import GLVRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchedDataset(Dataset):
    """Wraps the PARAM_RECOVERY_MATCHED.pkl file.

    Returns (X_normalized, r, A_flat) per sample. We standardize r and A using
    the dataset-wide std so the MSE loss is scale-balanced.
    """

    def __init__(self, path: str, split: str = "train", val_frac: float = 0.1, seed: int = 0):
        with open(path, "rb") as f:
            d = pickle.load(f)
        X = d["raw_trajectories"]                 # use raw, un-normalized trajectories
        r = d["r"]
        A = d["A"]
        N = X.shape[0]

        # Train/val split (deterministic)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        n_val = int(N * val_frac)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        pick = train_idx if split == "train" else val_idx

        self.X = X[pick]
        self.r = r[pick]
        self.A = A[pick]

        # Normalize each sample to peak 1 (so the model sees comparable input
        # ranges; r and A are unaffected by this since they parameterize the
        # underlying ODE).
        peaks = self.X.max(axis=(1, 2), keepdims=True)
        peaks[peaks == 0] = 1.0
        self.X_norm = self.X / peaks
        # We don't store peaks per sample since the regression doesn't need
        # them — the integration at inference time produces its own scale.

        self.split = split
        print(f"[{split}] N={len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_norm[idx]).float(),
            torch.from_numpy(self.r[idx]).float(),
            torch.from_numpy(self.A[idx].reshape(-1)).float(),
        )


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PARAM_RECOVERY_MATCHED.pkl")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save-route", default="model_ckpts/")
    ap.add_argument("--wandb-project", default="Conditional_LV_VAE_pivot")
    ap.add_argument("--no-log", dest="log", action="store_false", default=True)
    args = ap.parse_args()

    set_seed(args.seed)

    train_ds = MatchedDataset(args.data, split="train", seed=args.seed)
    val_ds = MatchedDataset(args.data, split="val", seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = GLVRegressor({"n_curves": 7, "seq_len": 65, "latent_dim": 30,
                          "rnn_hidden_size": 256, "rnn_num_layers": 2}).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.log:
        import wandb
        wandb.init(project=args.wandb_project, name=args.name,
                   config={**vars(args), "model": "glv-regression"},
                   job_type="train")

    best_val = float("inf")
    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        tloss = 0.0
        n_batches = 0
        for X, r_gt, A_flat_gt in train_dl:
            X = X.to(DEVICE, non_blocking=True)
            r_gt = r_gt.to(DEVICE, non_blocking=True)
            A_flat_gt = A_flat_gt.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            r_hat, A_hat = model.regress(X)
            r_loss = nn.functional.mse_loss(r_hat, r_gt)
            A_loss = nn.functional.mse_loss(A_hat.flatten(1), A_flat_gt)
            loss = r_loss + A_loss
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            n_batches += 1
        tloss /= max(n_batches, 1)

        # --- val ---
        model.eval()
        vloss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X, r_gt, A_flat_gt in val_dl:
                X = X.to(DEVICE, non_blocking=True)
                r_gt = r_gt.to(DEVICE, non_blocking=True)
                A_flat_gt = A_flat_gt.to(DEVICE, non_blocking=True)
                r_hat, A_hat = model.regress(X)
                r_loss = nn.functional.mse_loss(r_hat, r_gt)
                A_loss = nn.functional.mse_loss(A_hat.flatten(1), A_flat_gt)
                vloss += (r_loss + A_loss).item()
                n_batches += 1
        vloss /= max(n_batches, 1)

        scheduler.step()

        msg = f"epoch {epoch + 1:4d}/{args.epochs}  train={tloss:.5f}  val={vloss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}"
        print(msg)
        if args.log:
            import wandb
            wandb.log({"train_loss": tloss, "val_loss": vloss,
                       "lr": scheduler.get_last_lr()[0], "epoch": epoch})

        if vloss < best_val:
            best_val = vloss
            # Cache the empirical distribution into the saved checkpoint
            x0 = train_ds.X[:, :, 0]
            model.store_empirical_distribution(train_ds.r, train_ds.A, x0)
            ckpt = {
                "state_dict": model.state_dict(),
                "emp_r": model._emp_r,
                "emp_A": model._emp_A,
                "emp_x0": model._emp_x0,
                "config": {"n_curves": 7, "seq_len": 65, "latent_dim": 30,
                           "rnn_hidden_size": 256, "rnn_num_layers": 2},
            }
            out_path = Path(args.save_route) / f"{args.name}.pth"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out_path)

    if args.log:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
