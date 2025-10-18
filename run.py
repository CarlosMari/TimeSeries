import numpy as np
import matplotlib.pyplot as plt
from config import model_config, hp
from AE.model.autoencoder import Autoencoder
from train import train
from VAE.models.VAE import VAE
from VAE.models.CH_VAE import CHVAE
from VAE.models.MLP_VAE import MLPVAE
from VAE.models.AE import AE
import torch
from VAE.models.attention import create_memory_efficient_vae
from VAE.models.KAN import create_kan_vae
from VAE.models.CompleteKAN import KANCHVAE
from VAE.models.RNN import RecurrentVAE



torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#model = CHVAE(model_config)
#model = create_memory_efficient_vae(latent_dim=20)

# Configuration
config = {
    'in_channels': 7,
    'latent_dim': 32,
    # ... other config options
}

# Create model (use simplified=True for stable training)
#model = create_kan_vae(config, simplified=False)
model = RecurrentVAE(config)
# Use your existing train function
trained_model, _ = train(model, 'data/NEW_ORDERED_TRAIN_START.pkl')