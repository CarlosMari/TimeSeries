import numpy as np
import matplotlib.pyplot as plt
from config import model_config, hp
from AE.model.autoencoder import Autoencoder
from train import train
from VAE.models.VAE import VAE
from VAE.models.CH_VAE import CHVAE
from VAE.models.AE import AE
import torch


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = CHVAE(model_config)
trained_model, loss = train(model, 'data/TRAIN_NEW_DIST.pkl')