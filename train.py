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

DATA_TYPE = torch.float32
LOG = True 

TEST_ROUTE = 'data/MARIO_ORDERED_MAX_TEST.pkl'

#TEST_ROUTE = 'data/VAE_129_TRAIN.pkl'
np.random.seed(hp['random_seed'])

def load_data(data_route, batch_size):

    file = open(data_route,'rb')
    X = pickle.load(file)
    #X = np.loadtxt(data_route, delimiter = ",")

    # 0-1 Normalize the dataset
    X = (X - X.min())/(X.max() - X.min())
    #X = (X - X.min())/(3.0 - X.min())

    # Transfer it to torch Tensor

    X = torch.Tensor(X)
    data_loader = DataLoader(X, batch_size = hp["batch_size"], )
    return data_loader


def save_model(model, route):
    torch.save(model.state_dict(), route)


def inference(model, data_route, step, iters=10):

    model = model.eval()
    with open(data_route, 'rb') as file:
        X = pickle.load(file)

    X = (X - X.min()) / (X.max() - X.min())
    #X = (X - X.min()) / (3.0 - X.min())
    X = torch.Tensor(X)
    subset = X[0]  # shape: (7, 129)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = get_cmap('tab10')

    reconstructions = []
    for _ in range(iters):
        recons, _, _, log_var = model(subset.to(DEVICE).reshape(1, 7, 129))
        reconstructions.append(recons.cpu().detach())

    stacked = torch.stack(reconstructions, dim=0)  # [10, 1, 7, 129]
    reconstructions = stacked.squeeze(1).permute(1, 0, 2)  # [7, 10, 129]

    for i in range(subset.shape[0]):
        color = cmap(i / (subset.shape[0] - 1))
        x_vals = np.arange(subset.shape[1])

        # Original
        ax.plot(subset[i, :], color=color, label=f"Original {i}")

        # Mean & Std
        mean_recon = torch.mean(reconstructions[i], dim=0)
        std_recon = torch.std(reconstructions[i], dim=0)

        ax.plot(mean_recon, color=color, linestyle='--', label=f"Recon {i}")
        ax.fill_between(x_vals,
                        mean_recon - 2 * std_recon,
                        mean_recon + 2 * std_recon,
                        color=color,
                        alpha=0.2)

    ax.set_ylim([-0.1, 1])
    ax.set_title("Originals, Reconstructions (Mean ± 2σ)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

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


def test(model, data_route, step, num_samples=10):
    model = model.eval()
    criterion = torch.nn.MSELoss()
    data_loader = load_data(data_route, hp['batch_size'])

    total_loss = 0
    recon_loss = 0
    num_batches = 0
    total_in_interval = 0
    total_points = 0

    with torch.no_grad(): 
        for batch in data_loader:
            num_batches += 1
            batch = batch.to(DEVICE)

            # 1. Generate N reconstructions per input
            all_preds = []
            for _ in range(num_samples):
                pred, code, mu, log_var = model(batch)
                all_preds.append(pred.unsqueeze(0))  # shape: (1, X, 7, 129)

            preds_stack = torch.cat(all_preds, dim=0)  # shape: (N, X, 7, 129)

            # 2. Compute mean and std over the N samples
            mean_preds = preds_stack.mean(dim=0)  # (X, 7, 129)
            std_preds = preds_stack.std(dim=0)    # (X, 7, 129)

            # 3. Compute coverage: how often the original batch lies within mean ± 2σ
            lower = mean_preds - 2 * std_preds
            upper = mean_preds + 2 * std_preds
            in_interval = (batch >= lower) & (batch <= upper)  # (X, 7, 129)

            num_in_interval = in_interval.sum().item()
            num_total = batch.numel()
            total_in_interval += num_in_interval
            total_points += num_total

            # 4. Use one of the predictions to compute standard VAE loss (e.g., the first)
            pred = preds_stack[0]
            recon_loss += criterion(pred, batch)
            batch_loss, _, _ = model.loss(pred, batch, mu, log_var, code, hp["alpha"], len_dataset=len(data_loader.dataset))
            total_loss += batch_loss.item()

    # Compute final metrics
    coverage = total_in_interval / total_points

    # Log to wandb
    if LOG:
        wandb.log({
            'Eval_VAE_Loss': total_loss / num_batches,
            'Eval Recon Loss': recon_loss / num_batches,
            'Eval Coverage (mean±2σ)': coverage,
        }, step=step)




def train(model, data_route):

    model_config.update(hp, inplace=False)
    if LOG:
        wandb.init(
            project = 'Autoencoder VAEs',
            config = model_config,
            job_type = 'train',
        )
    
    model = model.to(DATA_TYPE).to(DEVICE).train()
    epochs = hp['epochs']
    lr = hp['lr']
    batch_size = hp['batch_size']
    data_loader = load_data(data_route, hp['batch_size'])

    running_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay= hp["weight_decay"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode='min', 
                                                 factor=0.95, 
                                                 patience=10, 
                                                 threshold=1e-4, 
                                                 min_lr=1e-6, 
                                                 cooldown = 10,)


    bar = tqdm(range(epochs))
    num_families = 0
    beta = 0
    beta_increment = 1 / (0.3 * (epochs-1))
    for i in bar:
        epoch_loss = 0 
        recon_losses = 0
        kl_losses = 0
        num_batches = 0
        for batch in data_loader:
            #print(f'{batch.shape=}')
            num_batches += 1
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            pred, code, mu, log_var = model(batch)

            batch_loss, recon_loss, kl_loss = model.loss(pred, batch, mu, log_var, code, hp["alpha"], len_dataset=len(data_loader.dataset), beta=beta)

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            recon_losses += recon_loss.item()
            kl_losses += kl_loss.item()

        # Log loss to wandb
        num_families += len(data_loader.dataset)
        if LOG:
            wandb.log({
                'Loss': epoch_loss/ num_batches,
                'Num Families': num_families,
                'KL Loss': kl_losses/ num_batches,
                'MSE Loss': recon_losses / num_batches,
                'Beta': beta,
                'lr': optimizer.param_groups[0]['lr'],

            }, step = i)

        running_losses.append(epoch_loss)

        if i % 5 == 0:
            model = model.eval()
            test(model, TEST_ROUTE, i)
            model = model.train()
        if i % 100 == 0:
            inference(model, TEST_ROUTE, i)

        if i <= int(0.3 * epochs):
            beta += beta_increment

        #else:
            #scheduler.step(recon_losses)


    inference(model, TEST_ROUTE, epochs)
    if LOG:
        wandb.finish()

    if model_config['save']:
        save_model(model, f'{model_config['save_route']}{model_config['name']}.pth')

    return model, running_losses




