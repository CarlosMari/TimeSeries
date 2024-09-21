import torch
import wandb
import numpy as np
from config import hp
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_TYPE = torch.float32
DEVICE = 'mps'


def load_data(data_route, batch_size):
    X = np.loadtxt(data_route, delimiter = ",")

    # 0-1 Normalize the dataset
    X = (X - X.min())/(X.max() - X.min())

    # Transfer it to torch Tensor
    X = torch.Tensor(X)
    X = X.reshape(( X.shape[0], 1, -1)).to(DATA_TYPE)

    data_loader = DataLoader(X, batch_size = hp["batch_size"], )
    return data_loader


def train(model, data_route):


    wandb.init(
        project = 'Autoencoder VAEs',
        config = hp,
        job_type = 'train',
    )
    
    model = model.to(DATA_TYPE).to(DEVICE)
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
        for batch in data_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            pred, code = model(batch)

            batch_loss = criterion(pred,batch)

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        # Log loss to wandb
        wandb.log({
            'Loss': epoch_loss
        }, step = i)

        running_losses.append(epoch_loss)

    return model, running_losses