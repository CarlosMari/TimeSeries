import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap

# --- Configuration ---
from src.utils.config import hp, model_config, DEVICE
# --- MODIFIED: Import the new LSTM_VAE model ---
# Make sure your new model file is accessible, e.g., in src/models/cvae.py
from src.models.cvae import LSTM_VAE 
from src.models.scale_contract import (
    LEGACY_SCALE_CONTRACT, SPECIES_SCALE_CONTRACT, scale_reconstruction_loss,
    validate_dataset_scale_contract,
)
from src.utils.checkpoints import make_checkpoint, write_checkpoint_payload

DATA_TYPE = torch.float32
LOG = True

# --- Set file paths ---
TRAIN_ROUTE = 'data/TRAIN_FINAL_PROCESSED.pkl'
TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'

np.random.seed(hp['random_seed'])

# --- 1. MODIFIED: Custom Dataset for the new data format ---
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data and their corresponding max values."""
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            data_package = pickle.load(f)
        self.features = data_package['data']
        # The target for the parallel MLP head
        self.max_values = data_package['reconstruction_max_values']
        self.sort_applied = data_package.get("sort_applied")
        if self.sort_applied is not None:
            self.sort_applied = bool(self.sort_applied)
        print(f"Loaded data from {file_path}. Num samples: {len(self.features)}")
        assert len(self.features) == len(self.max_values), "Data and max_value counts do not match."

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """Returns the normalized curve and its true max value."""
        return (
            torch.tensor(self.features[idx], dtype=DATA_TYPE), 
            torch.tensor(self.max_values[idx], dtype=DATA_TYPE)
        )

# --- 2. NEW: Loss function for the new model ---
# Module-level flag for optional spectral-loss term (set by train_pivot.py).
# Default 0 keeps train_cvae.py v1-bit-identical.
SPECTRAL_LOSS_WEIGHT = 0.0


def _spectral_loss(x_hat, x):
    """Magnitude-MSE in the FFT domain along the time axis.

    x_hat, x : (N, C, T). We FFT along T, take magnitudes, and MSE them. The
    DC term is excluded (covered by the regular recon MSE; including it just
    doubles the mean penalty).

    Note: cuFFT under AMP fp16 only supports power-of-2 sizes; T=65 is not,
    so we cast to float32 before the FFT.
    """
    # rfft along the last dim — force fp32 (cuFFT half precision needs power-of-2)
    Xh = torch.fft.rfft(x_hat.float(), dim=-1)
    X = torch.fft.rfft(x.float(), dim=-1)
    mag_diff = torch.abs(Xh).pow(2) - torch.abs(X).pow(2)
    # drop DC (index 0)
    return (mag_diff[..., 1:] ** 2).mean()


def vae_loss(x_hat, x, mu, log_var, max_vals_pred, max_vals_true, beta, lambda_max_val,
             scale_contract=LEGACY_SCALE_CONTRACT, max_vals_pred_transformed=None,
             scale_log_floor=1e-8):
    """Calculates the loss for the VAE with a parallel max value predictor.

    Args:
        max_vals_pred: Predicted max values in ORIGINAL space (after inverse transform) - Shape: (N, 7)
        max_vals_true: Ground truth max values in original space - Shape: (N, 7)

    Note: Loss is computed in ORIGINAL space for better reconstruction quality.
          The model internally predicts in transformed space (log), but we
          convert back to original space before computing loss.

          The historical legacy_sorted_v1 contract scores only curves 1-6.
          species_v2 instead uses all-species centered log-ratio error and requires
          the model's transformed head output. Its scale loss is in different units;
          lambda_max_val must be validated rather than borrowed as a tuned value.

          If SPECTRAL_LOSS_WEIGHT > 0, an FFT-magnitude MSE term is added.
          This is the post-pivot 2026-05-17 experiment to test whether
          explicitly penalizing spectral mismatch closes the chaos-diagnostics
          gap that all 7 architectures hit identically.

          M8 heteroscedastic-decoder convention: if x_hat has 2× the channels
          of x, the second half is interpreted as raw_log_σ (pre-softplus). We
          split, apply softplus + floor (matching HeteroscedasticLSTMVAE), and
          compute Gaussian NLL instead of MSE.
    """
    if x_hat.shape[1] == 2 * x.shape[1]:
        n_curves = x.shape[1]
        x_mu = x_hat[:, :n_curves, :]
        raw_log_sigma = x_hat[:, n_curves:, :]
        sigma = nn.functional.softplus(raw_log_sigma) + 1e-3
        recon_loss = (0.5 * ((x - x_mu) / sigma) ** 2 + torch.log(sigma)).mean()
    else:
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    max_val_loss = scale_reconstruction_loss(
        max_vals_pred, max_vals_true, scale_contract,
        transformed=max_vals_pred_transformed, log_floor=scale_log_floor,
    )

    total_loss = recon_loss + (beta * kl_divergence) + (lambda_max_val * max_val_loss)

    if SPECTRAL_LOSS_WEIGHT > 0.0:
        spec_loss = _spectral_loss(x_hat, x)
        total_loss = total_loss + SPECTRAL_LOSS_WEIGHT * spec_loss

    return total_loss, recon_loss, kl_divergence, max_val_loss

# --- 3. UPDATED: Inference and Plotting Functions ---

def inference_reconstruction(model, test_dataset, step, device=DEVICE, iters=5):
    """Performs reconstruction and plots normalized and de-normalized results."""
    model.eval()
    indices_to_plot = [5, 59]
    samples = [(test_dataset[i][0], test_dataset[i][1]) for i in indices_to_plot] # (X, max_vals)
    fig, axes = plt.subplots(2, len(samples), figsize=(18, 10), constrained_layout=True)
    if len(samples) == 1: axes = np.expand_dims(axes, axis=1)
    cmap = get_cmap('tab10')

    with torch.no_grad():
        for plot_idx, (X_cpu, max_vals_cpu) in enumerate(samples):
            single_X_gpu = X_cpu.to(device).unsqueeze(0)
            
            batched_X_gpu = single_X_gpu.expand(iters, -1, -1)
            batched_max_vals_gpu = max_vals_cpu.to(device).unsqueeze(0).expand(iters, -1)
            recons_norm_gpu, _, _, max_vals_pred_gpu,_ = model(batched_X_gpu, batched_max_vals_gpu, teacher_forcing_ratio=0)
            # M8 heteroscedastic: forward returns (N, 2C, T) — keep μ channels only for plotting.
            n_curves_plot = X_cpu.shape[0]
            if recons_norm_gpu.shape[1] == 2 * n_curves_plot:
                recons_norm_gpu = recons_norm_gpu[:, :n_curves_plot, :]

            original_denorm_cpu = X_cpu * max_vals_cpu.unsqueeze(-1)
            
            # --- FIXED: Use PREDICTED scale, not true scale ---
            # This shows what the model actually reconstructed using its predicted max values
            recons_denorm_gpu = recons_norm_gpu * max_vals_pred_gpu.unsqueeze(-1)
            # ------------------------------------

            # --- Top Row: Normalized Plot ---
            ax_norm = axes[0, plot_idx]
            for i in range(X_cpu.shape[0]):
                color, x_vals = cmap(i), np.arange(X_cpu.shape[1])
                ax_norm.plot(X_cpu[i, :], color=color, label=f"Original {i}")
                mean_recon = torch.mean(recons_norm_gpu[:, i, :], dim=0).cpu()
                std_recon = torch.std(recons_norm_gpu[:, i, :], dim=0).cpu()
                ax_norm.plot(x_vals, mean_recon, color=color, linestyle='--')
                ax_norm.fill_between(x_vals, mean_recon - 2*std_recon, mean_recon + 2*std_recon, color=color, alpha=0.2)
            ax_norm.set_ylim([-0.1, 1.1]); ax_norm.set_title(f"Normalized Recon (Sample {indices_to_plot[plot_idx]})"); ax_norm.legend()

            # --- Bottom Row: De-normalized Plot ---
            ax_denorm = axes[1, plot_idx]
            max_val_pred_mean = torch.mean(max_vals_pred_gpu, dim=0).cpu()
            for i in range(original_denorm_cpu.shape[0]):
                color, x_vals = cmap(i), np.arange(original_denorm_cpu.shape[1])
                ax_denorm.plot(x_vals, original_denorm_cpu[i, :], color=color)
                mean_recon_denorm = torch.mean(recons_denorm_gpu[:, i, :], dim=0).cpu()
                std_recon_denorm = torch.std(recons_denorm_gpu[:, i, :], dim=0).cpu()
                ax_denorm.plot(x_vals, mean_recon_denorm, color=color, linestyle='--')
                ax_denorm.fill_between(x_vals, mean_recon_denorm - 2*std_recon_denorm, mean_recon_denorm + 2*std_recon_denorm, color=color, alpha=0.2)
            title_str = f"De-normalized Recon\nMax Vals (True/Pred): " + ", ".join([f"{t:.2f}/{p:.2f}" for t, p in zip(max_vals_cpu, max_val_pred_mean)])
            ax_denorm.set_title(title_str, fontsize=9)

    fig.suptitle(f"Reconstruction Quality at Step {step}", fontsize=16)
    if LOG: wandb.log({"Reconstruction Plots": wandb.Image(fig)}, step=step)
    plt.close(fig)

def generate_unconditionally(model, step, num_samples=4, device=DEVICE):
    """Generates curves unconditionally and plots normalized and re-scaled results."""
    model.eval()
    fig, axes = plt.subplots(2, num_samples, figsize=(18, 9), constrained_layout=True)
    if num_samples == 1: axes = np.expand_dims(axes, axis=1)

    with torch.no_grad():
        generated_norm, predicted_max_vals = model.generate(num_samples, device)
        generated_denorm = generated_norm * predicted_max_vals.unsqueeze(-1)
        
        for j in range(num_samples):
            # --- Top Row: Normalized ---
            ax_norm = axes[0, j]
            curves_norm = generated_norm[j].cpu()
            for k in range(curves_norm.shape[0]):
                ax_norm.plot(curves_norm[k], label=f'Curve {k}')
            ax_norm.set_title(f"Generated Normalized (Sample {j+1})")
            ax_norm.set_ylim([-0.1, 1.1])
            
            # --- Bottom Row: De-normalized ---
            ax_denorm = axes[1, j]
            curves_denorm = generated_denorm[j].cpu()
            for k in range(curves_denorm.shape[0]):
                ax_denorm.plot(curves_denorm[k])
            pred_maxes_str = "Pred Maxes: " + ", ".join([f"{v:.2f}" for v in predicted_max_vals[j]])
            ax_denorm.set_title(pred_maxes_str, fontsize=9)
            
    fig.suptitle(f'Unconditionally Generated Curves at Step {step}', fontsize=16)
    if LOG: wandb.log({"Unconditional Generation": wandb.Image(fig)}, step=step)
    plt.close(fig)

def evaluate(model, test_loader, beta, lambda_max_val, device, num_samples=10):
    """
    Runs a full evaluation loop on the test dataset, including loss and coverage metrics.
    
    Args:
        model (nn.Module): The VAE model.
        test_loader (DataLoader): DataLoader for the test set.
        beta (float): The current KL divergence weight.
        lambda_max_val (float): The weight for the max value prediction loss.
        device (torch.device): The device to run on.
        num_samples (int): The number of times to sample for each input to calculate coverage.

    Returns:
        dict: A dictionary containing the averaged test losses and coverage statistics.
    """
    model.eval()
    
    # --- Loss Accumulators ---
    total_test_loss, recon_test_loss, kl_test_loss, max_val_test_loss = 0, 0, 0, 0
    
    # --- Coverage Accumulators ---
    total_in_interval = 0
    total_points = 0
    # Ensure accumulators are on the correct device
    per_curve_in_interval = torch.zeros(model.n_curves, device=device)
    per_curve_total_points = 0

    with torch.no_grad():
        for batch_X, batch_max_vals in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_max_vals = batch_max_vals.to(device, non_blocking=True)
            
            # --- Multi-pass for Coverage and Stable Loss Calculation ---
            all_preds = []
            for _ in range(num_samples):
                # We only need the reconstructed curve for coverage.
                # The other outputs (mu, log_var, max_vals_pred) are captured once outside the loop for efficiency.
                pred_sample, _, _, _, _ = model(batch_X, batch_max_vals, teacher_forcing_ratio=0)
                all_preds.append(pred_sample.unsqueeze(0))

            # Stack predictions along a new dimension: (num_samples, N, C, L)
            preds_stack = torch.cat(all_preds, dim=0)

            # Calculate statistics over the 'num_samples' dimension
            mean_preds = preds_stack.mean(dim=0)
            std_preds = preds_stack.std(dim=0)

            # --- Calculate Loss using the mean prediction for stability ---
            # We still need mu, log_var, etc. from a single forward pass
            _, mu, log_var, max_vals_pred, max_vals_pred_transformed = model(batch_X, batch_max_vals, teacher_forcing_ratio=0)
            
            loss, recon, kl, max_val = vae_loss(
                mean_preds, batch_X, mu, log_var, max_vals_pred, batch_max_vals, beta, lambda_max_val,
                scale_contract=getattr(model, "scale_contract", LEGACY_SCALE_CONTRACT),
                max_vals_pred_transformed=max_vals_pred_transformed,
                scale_log_floor=getattr(model, "scale_log_floor", 1e-8),
            )
            total_test_loss += loss.item()
            recon_test_loss += recon.item()
            kl_test_loss += kl.item()
            max_val_test_loss += max_val.item()

            # --- Calculate Coverage ---
            # M8 heteroscedastic: mean_preds/std_preds carry 2C channels — keep μ half only.
            cov_n_curves = batch_X.shape[1]
            cov_mean = mean_preds[:, :cov_n_curves, :] if mean_preds.shape[1] == 2 * cov_n_curves else mean_preds
            cov_std = std_preds[:, :cov_n_curves, :] if std_preds.shape[1] == 2 * cov_n_curves else std_preds
            lower_bound = cov_mean - 2 * cov_std
            upper_bound = cov_mean + 2 * cov_std

            in_interval = (batch_X >= lower_bound) & (batch_X <= upper_bound)
            
            # Overall coverage
            total_in_interval += in_interval.sum().item()
            total_points += batch_X.numel()

            # Per-curve coverage
            # Sum across batch and time dimensions (0 and 2), leaving the curve dimension (1)
            per_curve_in_interval += in_interval.sum(dim=(0, 2))
            
            # On the first batch, calculate the total points per curve for the whole dataset
            if per_curve_total_points == 0:
                num_total_samples = len(test_loader.dataset)
                seq_len = batch_X.shape[2]
                per_curve_total_points = num_total_samples * seq_len


    # --- Finalize Metrics ---
    num_batches = len(test_loader)
    
    # Finalize average losses
    avg_losses = {
        'Test Loss': total_test_loss / num_batches,
        'Test Recon Loss': recon_test_loss / num_batches,
        'Test KL Loss': kl_test_loss / num_batches,
        'Test Max Val Loss': max_val_test_loss / num_batches
    }

    # Finalize coverage metrics
    overall_coverage = total_in_interval / total_points if total_points > 0 else 0
    per_curve_coverage = (per_curve_in_interval / per_curve_total_points).cpu().tolist() if per_curve_total_points > 0 else [0] * model.n_curves

    coverage_metrics = {'Test Overall Coverage': overall_coverage}
    # Create a nice dictionary for logging per-curve coverage to wandb
    for i, cov in enumerate(per_curve_coverage):
        coverage_metrics[f'Test Coverage Curve {i}'] = cov

    # Combine all metrics into one dictionary to return
    avg_losses.update(coverage_metrics)
    return avg_losses

def train(model):
    corrected = getattr(model, "scale_contract", LEGACY_SCALE_CONTRACT) == SPECIES_SCALE_CONTRACT
    checkpoint_metadata = None
    checkpoint_path = None
    if corrected:
        checkpoint_metadata = {
            "seed": hp["random_seed"], "epochs": hp["epochs"],
            "data_train": TRAIN_ROUTE, "data_test": TEST_ROUTE,
            "hyperparameters": dict(hp),
        }
        # Validate JSON-safe metadata before W&B, device transfer, or fitting.
        make_checkpoint(model, checkpoint_metadata)
        if model_config["save"]:
            checkpoint_path = Path(model_config["save_route"]) / f'{model_config["name"]}.pth'
            if checkpoint_path.exists() or checkpoint_path.with_suffix(".progress.pth").exists():
                raise FileExistsError("Use a new corrected run name; checkpoint/progress file already exists")
    if LOG:
        # Check if wandb is already initialized (e.g., from sweep script)
        if wandb.run is None:
            # Properly combine hyperparameters and model config
            wandb.init(
                project='Conditional_LV_VAE',
                config={**hp, **model_config},
                job_type='train'
            )
    
    epochs = hp['epochs']
    lr = hp['lr']
    # --- NEW: Hyperparameter for max value prediction loss ---
    lambda_max_val = hp['lambda_max_val']

    # --- Use the new Dataset ---
    train_dataset = TimeSeriesDataset(TRAIN_ROUTE)
    test_dataset = TimeSeriesDataset(TEST_ROUTE)
    for dataset in (train_dataset, test_dataset):
        validate_dataset_scale_contract(
            torch.as_tensor(dataset.max_values),
            getattr(model, "scale_contract", LEGACY_SCALE_CONTRACT),
            sort_applied=dataset.sort_applied,
            allow_legacy_on_unsorted=model.config.get("allow_legacy_on_unsorted", False),
            n_curves=model.n_curves,
            normalized_curves=torch.as_tensor(dataset.features),
        )
    if corrected:
        checkpoint_metadata.update(sort_applied_train=train_dataset.sort_applied,
                                   sort_applied_test=test_dataset.sort_applied)
        if checkpoint_path is not None:
            write_checkpoint_payload(
                checkpoint_path.with_suffix(".progress.pth"),
                make_checkpoint(model, {**checkpoint_metadata, "epoch_completed": 0, "status": "initialized"}),
            )
    model = model.to(DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=hp["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=hp["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    bar = tqdm(range(epochs))
    beta_max = hp['beta_max']
    beta = 0.0  # warmup starts at 0; updated each epoch from the schedule below
    initial_tf_ratio, final_tf_ratio = 1.0, 0.025
    tf_decay_epochs = max(1, int(epochs * 0.4))
    warmup_epochs = max(1, hp.get('warmup_epochs', int(0.3 * epochs)))

    # --- Initial Evaluation ---
    generate_unconditionally(model, step=0, device=DEVICE)
    inference_reconstruction(model, test_dataset, step=0, device=DEVICE)
    
    for i in bar:
        # --- Update Schedules ---
        teacher_forcing_ratio = max(final_tf_ratio, initial_tf_ratio - (initial_tf_ratio - final_tf_ratio) * (i / tf_decay_epochs))
        beta = min(beta_max, beta_max * (i / warmup_epochs))

        # --- Training Loop ---
        model.train()
        epoch_loss, epoch_recon, epoch_kl, epoch_max_val = 0, 0, 0, 0
        
        for batch_X, batch_max_vals in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_max_vals = batch_max_vals.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                pred, mu, log_var, max_vals_pred, max_vals_pred_transform = model(batch_X, batch_max_vals, teacher_forcing_ratio=teacher_forcing_ratio)
                loss, recon, kl, max_val = vae_loss(
                    pred, batch_X, mu, log_var, max_vals_pred, batch_max_vals, beta, lambda_max_val,
                    scale_contract=getattr(model, "scale_contract", LEGACY_SCALE_CONTRACT),
                    max_vals_pred_transformed=max_vals_pred_transform,
                    scale_log_floor=getattr(model, "scale_log_floor", 1e-8),
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            epoch_max_val += max_val.item()
        
        # --- Logging ---
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches

        # Update progress bar with current metrics
        bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'recon': f'{avg_recon:.4f}',
            'beta': f'{beta:.2e}',
            'tf': f'{teacher_forcing_ratio:.3f}'
        })

        if LOG:
            wandb.log({
                'Train Loss': avg_loss,
                'Train Recon Loss': avg_recon,
                'Train KL Loss': epoch_kl / num_batches,
                'Train Max Val Loss': epoch_max_val / num_batches,
                'Beta': beta,
                'Teacher Forcing Ratio': teacher_forcing_ratio,
                'Epoch': i,
            }, step=i)
        
        # --- Evaluation every 20 epochs ---
        if (i > 0 and i % 20 == 0) or (i == epochs - 1):
            if checkpoint_path is not None:
                write_checkpoint_payload(
                    checkpoint_path.with_suffix(".progress.pth"),
                    make_checkpoint(model, {**checkpoint_metadata, "epoch_completed": i + 1, "status": "weight_recovery"}),
                    overwrite=True,
                )
            test_losses = evaluate(model, test_loader, beta, lambda_max_val, DEVICE)
            if LOG:
                wandb.log(test_losses, step=i)
            
            generate_unconditionally(model, step=i, device=DEVICE)
            inference_reconstruction(model, test_dataset, step=i, device=DEVICE)

    if LOG:
        wandb.finish()

    if model_config['save']:
        if corrected:
            payload = make_checkpoint(model, {**checkpoint_metadata, "epoch_completed": epochs, "status": "completed"})
            write_checkpoint_payload(checkpoint_path, payload)
        else:
            torch.save(model.state_dict(), f'{model_config["save_route"]}{model_config["name"]}.pth')
    return model

if __name__ == "__main__":
    # --- Remove obsolete config parameters and instantiate the new model ---
    if 'fingerprint_dim' in model_config:
        del model_config['fingerprint_dim']
    
    vae_model = LSTM_VAE(config=model_config)
    train(vae_model)
