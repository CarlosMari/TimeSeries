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
from VAE.models.cvae import ConditionalRecurrentVAE

DATA_TYPE = torch.float32
LOG = True

# --- Set file paths ---

TRAIN_ROUTE = 'data/TRAIN_PREPROCESSED_DS.pkl'
TEST_ROUTE = 'data/TEST_PREPROCESSED_DS.pkl'

np.random.seed(hp['random_seed'])

# --- 1. Custom Dataset for Labeled Data ---
class LabeledTimeSeriesDataset(Dataset):
    """PyTorch Dataset for features (curves) and labels (fingerprints)."""
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            data_package = pickle.load(f)
        self.features = data_package['data']
        self.labels = data_package['labels']
        print(f"Loaded data from {file_path}. Num samples: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=DATA_TYPE), torch.tensor(self.labels[idx], dtype=DATA_TYPE)


def inference_reconstruction(model, test_dataset, step, device=DEVICE, iters=10):
    """Performs reconstruction inference on specific test samples and logs the plot."""
    model.eval()
    indices_to_plot = [5, 59]
    samples = [(test_dataset[i][0], test_dataset[i][1]) for i in indices_to_plot]
    fig, axes = plt.subplots(1, len(samples), figsize=(18, 6), constrained_layout=True)
    if len(samples) == 1: axes = [axes]
    cmap = get_cmap('tab10')

    with torch.no_grad():
        for plot_idx, (subset_X_cpu, subset_y_cpu) in enumerate(samples):
            ax = axes[plot_idx]
            single_X_gpu = subset_X_cpu.to(device).unsqueeze(0)
            single_y_gpu = subset_y_cpu.squeeze().to(device).unsqueeze(0)
            batched_X_gpu = single_X_gpu.expand(iters, -1, -1)
            batched_y_gpu = single_y_gpu.expand(iters, -1)
            recons_batch_gpu, _, _, _ = model(batched_X_gpu, batched_y_gpu, teacher_forcing_ratio=0)
            reconstructions_cpu = recons_batch_gpu.permute(1, 0, 2).cpu()
            
            for i in range(subset_X_cpu.shape[0]):
                color = cmap(i)
                x_vals = np.arange(subset_X_cpu.shape[1])
                ax.plot(subset_X_cpu[i, :], color=color, label=f"Original {i}")
                recons_for_species_i = reconstructions_cpu[i]
                mean_recon = torch.mean(recons_for_species_i, dim=0)
                std_recon = torch.std(recons_for_species_i, dim=0)
                ax.plot(mean_recon, color=color, linestyle='--')
                ax.fill_between(x_vals, mean_recon - 2 * std_recon, mean_recon + 2 * std_recon, color=color, alpha=0.2)
            
            ax.set_ylim([-0.1, 1.1])
            ax.set_title(f"Reconstruction of Test Sample {indices_to_plot[plot_idx]}")
            ax.legend()

    fig.suptitle(f"Reconstruction Quality at Step {step}", fontsize=16)
    if LOG:
        wandb.log({"Reconstruction Plots": wandb.Image(fig)}, step=step)
    plt.close(fig)


def cvae_loss(x_hat, x, mu, log_var, y_hat, y, beta, gamma):
    """Calculates the loss for a Conditional VAE."""
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    param_loss = nn.functional.mse_loss(y_hat.squeeze(), y.squeeze(), reduction='mean')
    total_loss = recon_loss + beta * kl_divergence + gamma * param_loss
    return total_loss, recon_loss, kl_divergence, param_loss


def generate_with_condition(model, step, device=DEVICE):
    """Generates curves for specific behavioral fingerprints and logs them."""
    model.eval()
    fingerprints_to_test = {
        "Stable": torch.tensor([0.0, 0.0, 0.0], device=device),
        "Oscillating": torch.tensor([1.0, 1.0, 0.0], device=device),
        "Capacitor-Like": torch.tensor([0.5, 0.0, 1.0], device=device),
    }
    num_samples_per_type = 2
    fig, axes = plt.subplots(num_samples_per_type, len(fingerprints_to_test), 
                             figsize=(18, 8), constrained_layout=True)
    fig.suptitle(f'Conditionally Generated Curves at Step {step}', fontsize=16)
    
    with torch.no_grad():
        for i, (name, fingerprint) in enumerate(fingerprints_to_test.items()):
            generated_curves = model.generate(fingerprint, num_samples_per_type, device)
            for j in range(num_samples_per_type):
                ax = axes[j, i]
                curves_to_plot = generated_curves[j].cpu()
                for k in range(curves_to_plot.shape[0]):
                    ax.plot(curves_to_plot[k], label=f'Curve {k}')
                ax.set_title(f"{name} (Sample {j+1})")
                ax.set_ylim([-0.1, 1.1])

    if LOG:
        wandb.log({"Conditional Generation": wandb.Image(fig)}, step=step)
    plt.close(fig)

# --- NEW: Evaluation Function ---
def evaluate(model, test_loader, beta, gamma, device):
    """
    Runs a full evaluation loop on the test dataset.

    Args:
        model (nn.Module): The trained VAE model.
        test_loader (DataLoader): DataLoader for the test set.
        beta (float): The current KL divergence weight.
        gamma (float): The current fingerprint loss weight.
        device (torch.device): The device to run on (e.g., 'cuda').

    Returns:
        dict: A dictionary containing the averaged test losses.
    """
    model.eval()  # Set the model to evaluation mode
    
    total_test_loss = 0
    recon_test_loss = 0
    kl_test_loss = 0
    param_test_loss = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Run the forward pass with teacher_forcing_ratio=0 for a true test
            pred, mu, log_var, y_hat = model(batch_X, batch_y, teacher_forcing_ratio=0)
            
            # Calculate the loss for this batch
            loss, recon, kl, param = cvae_loss(
                pred, batch_X, mu, log_var, y_hat, batch_y, beta, gamma
            )

            # Accumulate the losses
            total_test_loss += loss.item()
            recon_test_loss += recon.item()
            kl_test_loss += kl.item()
            param_test_loss += param.item()
            
    # Calculate the average loss across all test batches
    num_batches = len(test_loader)
    avg_total_loss = total_test_loss / num_batches
    avg_recon_loss = recon_test_loss / num_batches
    avg_kl_loss = kl_test_loss / num_batches
    avg_param_loss = param_test_loss / num_batches

    # Return a dictionary for easy logging
    return {
        'Test Loss': avg_total_loss,
        'Test Recon Loss': avg_recon_loss,
        'Test KL Loss': avg_kl_loss,
        'Test Fingerprint Loss': avg_param_loss
    }


def train(model):
    if LOG:
        wandb.init(project='Conditional_LV_VAE', config=model.config, job_type='train')
    
    model = model.to(DEVICE)
    epochs = hp['epochs']
    lr = hp['lr']
    gamma = hp.get('gamma', 100.0)

    train_dataset = LabeledTimeSeriesDataset(TRAIN_ROUTE)
    test_dataset = LabeledTimeSeriesDataset(TEST_ROUTE)
    
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=hp["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=hp["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    bar = tqdm(range(epochs))
    beta = 0
    beta_max = hp['beta_max']
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.025
    tf_decay_epochs = int(epochs * 0.4)

    # --- Initial Evaluation ---
    generate_with_condition(model, step=0, device=DEVICE)
    inference_reconstruction(model, test_dataset, step=0)
    
    for i in bar:
        # --- Update Schedules ---
        if i < tf_decay_epochs:
            teacher_forcing_ratio = initial_tf_ratio - (initial_tf_ratio - final_tf_ratio) * (i / tf_decay_epochs)
        else:
            teacher_forcing_ratio = final_tf_ratio

        if i < hp.get('warmup_epochs', 0.3 * epochs):
            beta = beta_max * (i / hp.get('warmup_epochs', 0.3 * epochs))
        else:
            beta = beta_max

        # --- Training Loop ---
        model.train()
        epoch_loss, epoch_recon, epoch_kl, epoch_param = 0, 0, 0, 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                pred, mu, log_var, y_hat = model(batch_X, batch_y, teacher_forcing_ratio=teacher_forcing_ratio)
                loss, recon, kl, param = cvae_loss(pred, batch_X, mu, log_var, y_hat, batch_y, beta, gamma)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            epoch_param += param.item()
        
        # --- Logging ---
        num_batches = len(train_loader)
        if LOG:
            wandb.log({
                'Train Loss': epoch_loss / num_batches,
                'Recon Loss': epoch_recon / num_batches,
                'KL Loss': epoch_kl / num_batches,
                'Fingerprint Loss': epoch_param / num_batches,
                'Beta': beta,
                'Teacher Forcing Ratio': teacher_forcing_ratio,
                'Epoch': i,
            }, step=i)
        
        # --- Evaluation every 10 epochs ---
        if (i > 0 and i % 10 == 0) or (i == epochs - 1):
            # --- MODIFIED: Run full test set evaluation ---
            test_losses = evaluate(model, test_loader, beta, gamma, DEVICE)
            if LOG:
                # Log the dictionary of test losses to wandb
                wandb.log(test_losses, step=i)
            # --- END OF MODIFICATION ---
            
            # The rest of your evaluation plotting
            generate_with_condition(model, step=i, device=DEVICE)
            inference_reconstruction(model, test_dataset, step=i)

    if LOG:
        wandb.finish()

    if model_config['save']:
        torch.save(model.state_dict(), f'{model_config["save_route"]}{model_config["name"]}.pth')

    return model

if __name__ == "__main__":
    model_config['fingerprint_dim'] = 3
    cvae_model = ConditionalRecurrentVAE(config=model_config)
    train(cvae_model)
