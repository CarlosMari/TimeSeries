import torch
import wandb
import numpy as np
from config import hp, model_config
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import pickle
from sklearn.decomposition import PCA
from VAE.models.VAE import VAE


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

    _, latents, _, _ = model(X.to(DEVICE))
    recons, _, _, _ = model(subset)
    recons = recons.cpu().detach()
    # Transfer to cpu and drop gradients to enable plotting
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

    x_coord = latents[:, 0].cpu().detach().T
    y_coord = latents[:, 1].cpu().detach().T

    fig2, axs2 = plt.subplots()
    axs2.scatter(x_coord, y_coord, s=50, c="w", edgecolor="b")
    plt.title('Latent Dimension')

    if LOG:
        wandb.log({"plot": wandb.Image(fig),
                   "latent": wandb.Image(fig2)})
        

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

    # If number of batches is different than train change the code
    with torch.no_grad(): 
        for batch in data_loader:
            num_batches += 1
            batch = batch.to(DEVICE)
            pred, code, mu, log_var = model(batch)
            batch_loss = VAE.vae_loss(pred, batch, mu, log_var, hp["alpha"])
            total_loss += batch_loss.item()
    
    # Log loss to wandb
    if LOG:
        wandb.log({
            'Eval_Loss': total_loss/num_batches
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
    criterion = torch.nn.MSELoss()

    bar = tqdm(range(epochs))
    for i in bar:
        epoch_loss = 0 
        num_batches = 0
        for batch in data_loader:
            num_batches += 1
            batch = batch.to(DEVICE)

            #indices = get_random_indices(model_config).to(DEVICE)

            #batch = complete_batch[:, :, indices]
            #batch = complete_batch

            optimizer.zero_grad()
            #pred, code = model(batch)
            pred, code, mu, log_var = model(batch)

            #batch_loss = #criterion(pred,batch)
            batch_loss = VAE.vae_loss(pred, batch, mu, log_var, hp["alpha"])



            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        # Log loss to wandb
        if LOG:
            wandb.log({
                'Loss': epoch_loss/ num_batches
            }, step = i)

        running_losses.append(epoch_loss)

        if i % 5 == 0:
            model = model.eval()
            test(model, './data/VAE_129.pkl', i)
            model = model.train()


    inference(model, './data/VAE_129.pkl')
    if LOG:
        wandb.finish()


    if model_config['save']:
        save_model(model, f'{model_config['save_route']}{model_config['name']}.pth')

    return model, running_losses




