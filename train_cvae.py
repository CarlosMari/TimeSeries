import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap

# --- Configuration ---
from config import hp, model_config, DEVICE
# --- MODIFIED: Import the new LSTM_VAE model ---
# Make sure your new model file is accessible, e.g., in VAE/models/lstm_vae.py
from VAE.models.cvae import LSTM_VAE 

DATA_TYPE = torch.float32
LOG = True

# --- Set file paths ---
TRAIN_ROUTE = 'data/TRAIN_PREPROCESSED_DS.pkl'
TEST_ROUTE = 'data/TEST_PREPROCESSED_DS.pkl'

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
def vae_loss(x_hat, x, mu, log_var, max_vals_pred, max_vals_true, beta, lambda_max_val):
    """Calculates the loss for the VAE with a parallel max value predictor."""
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    max_val_loss = nn.functional.mse_loss(max_vals_pred, max_vals_true, reduction='mean')
    
    total_loss = recon_loss + (beta * kl_divergence) + (lambda_max_val * max_val_loss)
    
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
            recons_norm_gpu, _, _, max_vals_pred_gpu = model(batched_X_gpu, teacher_forcing_ratio=0)
            
            original_denorm_cpu = X_cpu * max_vals_cpu.unsqueeze(-1)
            
            # --- THIS IS THE CORRECTED LINE ---
            # Move max_vals_cpu to the correct device before multiplying
            recons_denorm_gpu = recons_norm_gpu * max_vals_cpu.to(device).unsqueeze(0).unsqueeze(-1)
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
                pred_sample, _, _, _ = model(batch_X, teacher_forcing_ratio=0)
                all_preds.append(pred_sample.unsqueeze(0))

            # Stack predictions along a new dimension: (num_samples, N, C, L)
            preds_stack = torch.cat(all_preds, dim=0)

            # Calculate statistics over the 'num_samples' dimension
            mean_preds = preds_stack.mean(dim=0)
            std_preds = preds_stack.std(dim=0)

            # --- Calculate Loss using the mean prediction for stability ---
            # We still need mu, log_var, etc. from a single forward pass
            _, mu, log_var, max_vals_pred = model(batch_X, teacher_forcing_ratio=0)
            
            loss, recon, kl, max_val = vae_loss(
                mean_preds, batch_X, mu, log_var, max_vals_pred, batch_max_vals, beta, lambda_max_val
            )
            total_test_loss += loss.item()
            recon_test_loss += recon.item()
            kl_test_loss += kl.item()
            max_val_test_loss += max_val.item()

            # --- Calculate Coverage ---
            lower_bound = mean_preds - 2 * std_preds
            upper_bound = mean_preds + 2 * std_preds
            
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
    if LOG:
        wandb.init(project='Conditional_LV_VAE', config=model.config, job_type='train')
    
    model = model.to(DEVICE)
    epochs = hp['epochs']
    lr = hp['lr']
    # --- NEW: Hyperparameter for max value prediction loss ---
    lambda_max_val = hp['lambda_max_val']

    # --- Use the new Dataset ---
    train_dataset = TimeSeriesDataset(TRAIN_ROUTE)
    test_dataset = TimeSeriesDataset(TEST_ROUTE)
    
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=hp["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=hp["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    bar = tqdm(range(epochs))
    #beta, beta_max = 0, hp['beta_max']
    beta, beta_max = hp['beta_max'], hp['beta_max']
    initial_tf_ratio, final_tf_ratio = 1.0, 0.025
    tf_decay_epochs = int(epochs * 0.4)
    warmup_epochs = hp.get('warmup_epochs', int(0.3 * epochs))

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
                pred, mu, log_var, max_vals_pred = model(batch_X, teacher_forcing_ratio=teacher_forcing_ratio)
                loss, recon, kl, max_val = vae_loss(
                    pred, batch_X, mu, log_var, max_vals_pred, batch_max_vals, beta, lambda_max_val
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
        if LOG:
            wandb.log({
                'Train Loss': epoch_loss / num_batches,
                'Train Recon Loss': epoch_recon / num_batches,
                'Train KL Loss': epoch_kl / num_batches,
                'Train Max Val Loss': epoch_max_val / num_batches,
                'Beta': beta,
                'Teacher Forcing Ratio': teacher_forcing_ratio,
                'Epoch': i,
            }, step=i)
        
        # --- Evaluation every 20 epochs ---
        if (i > 0 and i % 20 == 0) or (i == epochs - 1):
            test_losses = evaluate(model, test_loader, beta, lambda_max_val, DEVICE)
            if LOG:
                wandb.log(test_losses, step=i)
            
            generate_unconditionally(model, step=i, device=DEVICE)
            inference_reconstruction(model, test_dataset, step=i, device=DEVICE)

    if LOG:
        wandb.finish()

    if model_config['save']:
        torch.save(model.state_dict(), f'{model_config["save_route"]}{model_config["name"]}.pth')
    return model

if __name__ == "__main__":
    # --- Remove obsolete config parameters and instantiate the new model ---
    if 'fingerprint_dim' in model_config:
        del model_config['fingerprint_dim']
    
    vae_model = LSTM_VAE(config=model_config)
    train(vae_model)