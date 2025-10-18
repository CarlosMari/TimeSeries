import torch
import wandb
import numpy as np
from config import hp, model_config, DEVICE
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import pickle
from sklearn.decomposition import PCA
from VAE.models.VAE import VAE
import torch.optim as optim
from utils.loss import alpha_div_loss


DATA_TYPE = torch.float32
LOG = True

TEST_ROUTE = 'data/NEW_ORDERED_TEST_START.pkl'

#TEST_ROUTE = 'data/VAE_129_TRAIN.pkl'
np.random.seed(hp['random_seed'])

def load_data(data_route, batch_size):

    file = open(data_route,'rb')
    X = pickle.load(file)
    file.close()
    #X = np.loadtxt(data_route, delimiter = ",")

    # 0-1 Normalize the dataset
    #X = (X - X.min())/(X.max() - X.min())
    X_min = X.min(axis=(1, 2), keepdims=True)  # shape (N, 1, 1)
    X_max = X.max(axis=(1, 2), keepdims=True)  # shape (N, 1, 1)

    # Normalize each "family" (N index) individually
    X = (X - X_min) / (X_max - X_min)
    #X = (X - X.min())/(3.0 - X.min())

    # Transfer it to torch Tensor

    X = torch.Tensor(X)
    data_loader = DataLoader(X, batch_size = hp["batch_size"], )
    return data_loader


def save_model(model, route):
    torch.save(model.state_dict(), route)


def inference(model, subsets_to_plot, step, iters=10):
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    cmap = get_cmap('tab10')

    for plot_idx, subset in enumerate(subsets_to_plot):
        ax = axes[plot_idx]
        
        with torch.no_grad():
            # --- START OF THE OPTIMIZED SECTION ---

            # 1. Prepare the input: move to GPU and add a batch dimension
            single_subset_gpu = subset.to(DEVICE).unsqueeze(0)  # Shape: [1, 7, 129]

            # 2. Create a batch by expanding the single input `iters` times.
            #    The -1 tells expand to not change that dimension's size.
            batched_input = single_subset_gpu.expand(iters, -1, -1)  # Shape: [iters, 7, 129]

            # 3. Run the model just ONCE on the entire parallelized batch
            recons_batch, _, _, _ = model(batched_input)  # Output shape is [iters, 7, 129]

            # 4. Permute to get [channels, iters, length] for plotting and move to CPU
            reconstructions = recons_batch.permute(1, 0, 2).cpu()

            # --- END OF THE OPTIMIZED SECTION ---

        # The plotting logic remains exactly the same
        for i in range(subset.shape[0]):
            color = cmap(i / (subset.shape[0] - 1))
            x_vals = np.arange(subset.shape[1])

            # Original
            ax.plot(subset[i, :], color=color, label=f"Original {i}")

            # Mean & Std of Reconstructions
            mean_recon = torch.mean(reconstructions[i], dim=0)
            std_recon = torch.std(reconstructions[i], dim=0)

            ax.plot(mean_recon, color=color, linestyle='--', label=f"Recon {i}")
            ax.fill_between(x_vals,
                            mean_recon - 2 * std_recon,
                            mean_recon + 2 * std_recon,
                            color=color,
                            alpha=0.2)

        ax.set_ylim([-0.1, 1.1])
        ax.set_title(f"Subset {plot_idx}: Originals & Recon (Mean ± 2σ)")

    plt.tight_layout()

    if LOG:
        wandb.log({"plot": wandb.Image(fig)}, step=step)

    plt.close('all')


        

def get_random_indices(model_config):

    N = model_config["input_size"]
    p = model_config["sampling"]

    size = int(p * N)

    # Generate random indices
    indices = torch.randperm(N)[:size]

    # Sort the indices to maintain order
    indices, _ = torch.sort(indices)
    return indices


#
# --- THIS IS THE CORRECTED `test` FUNCTION ---
#
def test(model, data_loader, step, num_samples=10): # <-- CHANGE 1: Accepts a data_loader
    model = model.eval()
    criterion = torch.nn.MSELoss()

    total_loss = 0
    recon_loss_val = 0 # Renamed to avoid confusion with function argument
    num_batches = 0
    total_in_interval = 0
    total_points = 0

    per_curve_in_interval = torch.zeros(7).to(DEVICE)
    per_curve_total = torch.zeros(7).to(DEVICE)

    with torch.no_grad(): 
        # CHANGE 2: Removed the call to `load_data`
        for batch in data_loader:
            num_batches += 1
            batch = batch.to(DEVICE)

            # The rest of your function is fine
            all_preds = []
            for _ in range(num_samples):
                pred, code, mu, log_var = model(batch)
                all_preds.append(pred.unsqueeze(0))

            preds_stack = torch.cat(all_preds, dim=0)
            mean_preds = preds_stack.mean(dim=0)
            std_preds = preds_stack.std(dim=0)

            lower = mean_preds - 2 * std_preds
            upper = mean_preds + 2 * std_preds
            in_interval = (batch >= lower) & (batch <= upper)

            total_in_interval += in_interval.sum().item()
            total_points += batch.numel()

            per_curve_in_interval += in_interval.sum(dim=(0, 2))
            per_curve_total += torch.ones_like(batch).sum(dim=(0, 2))

            pred = preds_stack[0]
            recon_loss_val += criterion(pred, batch)
            # Make sure your model.loss doesn't require the 'z' (code) argument if it's not used
            # Based on my previous suggestion, it doesn't.
            batch_loss, _, _ = alpha_div_loss(pred, batch, mu, log_var,code,0, len_dataset=len(data_loader.dataset))
            total_loss += batch_loss.item()

    coverage = total_in_interval / total_points
    per_curve_coverage = (per_curve_in_interval / per_curve_total).cpu().tolist() # Added .cpu()

    if LOG:
        log_dict = {
            'Eval_VAE_Loss': total_loss / num_batches,
            'Eval Recon Loss': recon_loss_val / num_batches,
            'Eval Coverage (mean±2σ)': coverage,
        }
        for i, cov in enumerate(per_curve_coverage):
            log_dict[f'Eval Coverage Channel {i}'] = cov
        wandb.log(log_dict, step=step)


#
# --- NEW FUNCTION TO BE ADDED ---
#

def generate_and_log_curves(model, step, num_samples=4, device=DEVICE):
    """
    Generates new curves from random latent vectors and logs them to wandb.
    This function handles both convolutional and recurrent VAE architectures.
    """
    model.eval()
    print(f"\n--- Generating new curves at step {step} ---")

    with torch.no_grad():
        # 1. Sample random vectors from the prior distribution N(0, I)
        z = torch.randn(num_samples, model.config['latent_dim']).to(device)
        
        # 2. Decode the latent vectors into curves
        # The generation process depends on the model architecture.
        
        # Check if the model is recurrent (has an 'encoder_rnn' attribute)
        # --- Autoregressive Generation for RecurrentVAE --- 
        # Initialize hidden state from z
        h_0 = model.latent_to_hidden(z).view(model.rnn_num_layers, num_samples, model.rnn_hidden_size).contiguous()
        c_0 = model.latent_to_cell(z).view(model.rnn_num_layers, num_samples, model.rnn_hidden_size).contiguous()
        hidden_state = (h_0, c_0)
        
        # Start with a zero-tensor as the first input
        current_step_input = torch.zeros(num_samples, 1, model.n_curves).to(device)
        
        generated_steps = []
        for _ in range(model.seq_len):
            output, hidden_state = model.decoder_rnn(current_step_input, hidden_state)
            output = model.output_map(output)
            current_step_input = output # Use own output as next input
            generated_steps.append(output)

        generated_curves = torch.cat(generated_steps, dim=1).permute(0, 2, 1) # (N, L, C) -> (N, C, L)


        # Apply final clamp and move to CPU for plotting
        generated_curves = torch.clamp(generated_curves, min=0).cpu()

    # 3. Plotting the results
    # We'll create a 2x2 grid for the 4 samples.
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    cmap = get_cmap('viridis')

    for i in range(num_samples):
        ax = axes[i]
        curves_to_plot = generated_curves[i] # Shape: [7, 129]
        for j in range(curves_to_plot.shape[0]):
            color = cmap(j / curves_to_plot.shape[0])
            ax.plot(curves_to_plot[j], color=color, label=f'Curve {j}')
        ax.set_title(f'Generated Sample {i+1}')
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
    
    fig.suptitle(f'Generated Curves at Training Step {step}', fontsize=16)

    # 4. Log the plot to Weights & Biases
    if LOG:
        wandb.log({"Generated Curves": wandb.Image(fig)}, step=step)

    plt.close(fig) # Prevent plots from displaying in notebooks
    model.train() # Set model back to training mode


def train(model, data_route):

    model_config.update(hp, inplace=False)
    if LOG:
        wandb.init(
            project='Autoencoder VAEs',
            config=model_config,
            job_type='train',
        )
    
    model = model.to(DATA_TYPE).to(DEVICE).train()
    epochs = hp['epochs']
    lr = hp['lr']

    # --- FIX 1: Load data ONCE and configure DataLoader correctly ---
    # Load training data
    print(f'{data_route=}')
    with open(data_route, 'rb') as file:
        X_train = pickle.load(file)
    # Normalize (assuming the same logic as before)
    X_train_min = X_train.min(axis=(1, 2), keepdims=True)
    X_train_max = X_train.max(axis=(1, 2), keepdims=True)
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    
    train_dataset = torch.Tensor(X_train)
    # FIX 1.1: Use num_workers and pin_memory
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hp["batch_size"], 
        shuffle=True,
        num_workers=4,  # Use multiple processes for data loading
        pin_memory=True # Speeds up CPU->GPU transfer
    )

    # Load test data ONCE
    with open(TEST_ROUTE, 'rb') as file:
        X_test = pickle.load(file)
    X_test_min = X_test.min(axis=(1, 2), keepdims=True)
    X_test_max = X_test.max(axis=(1, 2), keepdims=True)
    X_test = (X_test - X_test_min) / (X_test_max - X_test_min)

    test_dataset = torch.Tensor(X_test)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=hp["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # Get fixed subsets for inference plotting
    inference_subsets = [test_dataset[23], test_dataset[45]]


    # --- Setup Optimizer and Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)
    
    # --- FIX 2: Use Automatic Mixed Precision (AMP) ---
    scaler = torch.cuda.amp.GradScaler()

    bar = tqdm(range(epochs))
    beta = 0
    beta_increment = 1 / (0.3 * (epochs - 1))

    # --- FIX 3: Lighter initial evaluation ---
    # Run a quick check before training starts
    # Note: Modify your functions to accept data instead of file paths
    inference(model, inference_subsets, 0)
    test(model, test_loader, 0)

    for i in bar:
        model.train() # Make sure model is in training mode
        epoch_loss = 0 
        recon_losses = 0
        kl_losses = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True) # non_blocking is good practice with pin_memory

            optimizer.zero_grad(set_to_none=True) # set_to_none is slightly faster

            # AMP: autocast context manager
            with torch.cuda.amp.autocast():
                pred, code, mu, log_var = model(batch)
                batch_loss, recon_loss, kl_loss = alpha_div_loss(
                    pred, batch, mu, log_var,code,0, len_dataset=len(train_loader.dataset), beta=beta
                )

            # AMP: Scale loss and call backward
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += batch_loss.item()
            recon_losses += recon_loss.item()
            kl_losses += kl_loss.item()
        
        num_batches = len(train_loader)
        if LOG:
            wandb.log({
                'Loss': epoch_loss / num_batches,
                'KL Loss': kl_losses / num_batches,
                'MSE Loss': recon_losses / num_batches,
                'Beta': beta,
                'lr': optimizer.param_groups[0]['lr'],
            }, step=i)
        
        # --- FIX 3: Lighter and less frequent evaluation ---
        # Evaluate every 50 epochs instead of 25 to reduce overhead
        if (i > 0 and i % 25 == 0) or (i == epochs - 1):
            model.eval()
            # Pass the pre-loaded data to avoid I/O
            # Consider using a single fixed batch for test() for speed
            test(model, test_loader, step=i) 
            inference(model, inference_subsets, step=i) # Pass subsets, not the file path
            generate_and_log_curves(model,step=i)
            model.train()

        if i <= int(0.3 * epochs):
            beta = min(1.0, beta + beta_increment) # Cap beta at 1.0

        #scheduler.step(recon_losses / num_batches)

    # Final full evaluation at the end
    inference(model, inference_subsets, epochs)
    
    if LOG:
        wandb.finish()

    if model_config['save']:
        save_model(model, f'{model_config["save_route"]}{model_config["name"]}.pth')

    return model



