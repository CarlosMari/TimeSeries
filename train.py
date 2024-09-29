import torch
import wandb
import numpy as np
from config import hp, model_config
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import pickle

DATA_TYPE = torch.float32
DEVICE = 'mps'
LOG = True



np.random.seed(hp['random_seed'])

def load_data(data_route, batch_size):

    file = open(data_route,'rb')
    X = pickle.load(file)
    #X = np.loadtxt(data_route, delimiter = ",")

    # 0-1 Normalize the dataset
    X = (X - X.min())/(X.max() - X.min())

    # Transfer it to torch Tensor
    X = torch.Tensor(X)
    X = X.reshape(( X.shape[0], 1, -1)).to(DATA_TYPE)

    data_loader = DataLoader(X, batch_size = hp["batch_size"], )
    return data_loader


def save_model(model, route):
    torch.save(model.state_dict(), route)

def inference(model, data_route):

    model = model.eval()
    file = open(data_route,'rb')
    X = pickle.load(file)
    #X = np.loadtxt(data_route, delimiter = ",")
    X = (X - X.min())/(X.max() - X.min())
    X = torch.Tensor(X)
    X = X.reshape(( X.shape[0], 1, -1)).to(DATA_TYPE)

    subset = X[np.random.randint(0,X.shape[0], size = 7), :,:].to(DEVICE)

    # Generate embeddings and reconstructions
    embeddings_sub = model.encoder(subset)
    recons = model.decoder(embeddings_sub).cpu().detach()

    # Transfer to cpu and drop gradients to enable plotting
    embeddings_sub = embeddings_sub.cpu().detach()
    subset = subset.cpu().detach()

    fig, axs = plt.subplots(1,2, figsize = (18,6))
    axs = axs.flatten()


    cmap = get_cmap('tab10')  # You can change 'tab10' to any other colormap

    # Plot originals
    for i in range(subset.shape[0]):
        axs[0].plot(subset[i].squeeze(), color=cmap(i))
    axs[0].set_title("Originals")
    #axs[0].set_xlim([0, 50])

    # Plot reconstructions
    for i in range(recons.shape[0]):
        axs[1].plot(recons[i].squeeze(), color=cmap(i))
    axs[1].set_title("Reconstructions")
    #axs[1].set_xlim([0, 50])

    plt.tight_layout()
    if LOG:
        wandb.log({"plot": wandb.Image(fig)})

def get_random_indices(model_config):

    N = model_config["input_size"]
    p = model_config["sampling"]

    size = int(p * N)

    # Generate random indices
    indices = torch.randperm(N)[:size]

    # Sort the indices to maintain order
    indices, _ = torch.sort(indices)
    return indices


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
    criterion = torch.nn.MSELoss()

    bar = tqdm(range(epochs))
    for i in bar:
        epoch_loss = 0 
        for complete_batch in data_loader:
            complete_batch = complete_batch.to(DEVICE)

            indices = get_random_indices(model_config).to(DEVICE)

            batch = complete_batch[:, :, indices]
            #batch = batch.to(DEVICE)

            optimizer.zero_grad()
            pred, code = model(batch)

            batch_loss = criterion(pred,complete_batch)

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        # Log loss to wandb
        if LOG:
            wandb.log({
                'Loss': epoch_loss
            }, step = i)

        running_losses.append(epoch_loss)


    inference(model, data_route)
    if LOG:
        wandb.finish()


    if model_config['save']:
        save_model(model, f'{model_config['save_route']}{model_config['name']}.pth')

    return model, running_losses




