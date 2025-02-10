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


DATA_TYPE = torch.float32
LOG = True 

TEST_ROUTE = 'data/TEST_NEW_DIST.pkl'

#TEST_ROUTE = 'data/VAE_129_TRAIN.pkl'
np.random.seed(hp['random_seed'])

def load_data(data_route, batch_size):

    file = open(data_route,'rb')
    X = pickle.load(file)
    #X = np.loadtxt(data_route, delimiter = ",")

    # 0-1 Normalize the dataset
    X = (X - X.min())/(X.max() - X.min())

    # Transfer it to torch Tensor

    X = torch.Tensor(X)
    data_loader = DataLoader(X, batch_size = hp["batch_size"], )
    return data_loader


def save_model(model, route):
    torch.save(model.state_dict(), route)

def inference(model, data_route, step, iters=10):

    model = model.eval()
    file = open(data_route,'rb')
    X = pickle.load(file)
    X = (X - X.min())/(X.max() - X.min())
    X = torch.Tensor(X)
    subset = X[0]


    fig, axs = plt.subplots(1,2, figsize = (18,6))
    axs = axs.flatten()
    cmap = get_cmap('tab10')

    reconstructions = []
    for z in range(iters):
        recons, _, _, log_var = model(subset.to(DEVICE).reshape(1,7,129))
        recons = recons.cpu().detach()
        subset = subset.cpu().detach()
        reconstructions.append(recons)

    stacked = torch.stack(reconstructions, dim=0)  # Shape: [10, 1, 7, 129]
    # Remove the batch dimension and transpose to get curves first
    reconstructions = stacked.squeeze(1).permute(1, 0, 2)
    # Plot originals
    for i in range(subset.shape[0]):
        color = cmap(i / (subset.shape[0] - 1))
        axs[0].plot(subset[i,:], color=color)
        
        for z in range(iters):
            axs[1].plot(reconstructions[i,z,:], color=color, alpha=0.1)
        mean_recon = torch.mean(reconstructions[i], dim=0)  # Average over iterations
        axs[1].plot(mean_recon, color=color, linestyle='--')

    axs[0].set_title("Originals")
    axs[1].set_title("Reconstructions")

    plt.tight_layout()

    if LOG:
        wandb.log({"plot": wandb.Image(fig),},
                  step=step)
                   #"latent": wandb.Image(fig2)})
    
    #plt.show()
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


def test(model, data_route, step):
    model = model.eval()
    criterion = torch.nn.MSELoss()
    data_loader = load_data(data_route, hp['batch_size'])


    total_loss = 0
    num_batches = 0
    recon_loss = 0
    # If number of batches is different than train change the code
    with torch.no_grad(): 
        for batch in data_loader:
            num_batches += 1
            batch = batch.to(DEVICE)
            pred, code, mu, log_var = model(batch)
            recon_loss += criterion(pred, batch)
            batch_loss, _, _ = model.loss(pred, batch, mu, log_var, code, hp["alpha"])
            total_loss += batch_loss.item()
    
    # Log loss to wandb
    if LOG:
        wandb.log({
            'Eval_VAE_Loss': total_loss/num_batches,
            'Eval Recon Loss':  recon_loss/num_batches,
        }, step = step)



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

    bar = tqdm(range(epochs))
    num_families = 0
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

            batch_loss, recon_loss, kl_loss = model.loss(pred, batch, mu, log_var, code, hp["alpha"])

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
                'MSE Loss': recon_loss / num_batches

            }, step = i)

        running_losses.append(epoch_loss)

        if i % 5 == 0:
            model = model.eval()
            test(model, TEST_ROUTE, i)
            model = model.train()
        if i % 100 == 0:
            inference(model, TEST_ROUTE, i)


    inference(model, TEST_ROUTE, epochs)
    if LOG:
        wandb.finish()

    if model_config['save']:
        save_model(model, f'{model_config['save_route']}{model_config['name']}.pth')

    return model, running_losses




