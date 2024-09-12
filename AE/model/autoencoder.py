import torch
import torch.nn as nn
from AE.model.CausalCNN import CausalCNNEncoder

class Autoencoder(nn.Module):
    """
    Autoencoder for timeseries
    
    -------------------
    Attributes:
        encoder: Encoder
            The encoder part of the AE
        decoder: nn.Sequential 
            Simple Multilayer Perceptron

    ------------------
    """
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config
        self.encoder = CausalCNNEncoder(
            in_channels=config['in_channels'],
            channels=config['channels'],
            depth=config['depth'],
            reduced_size=config['reduced_size'],
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size']
        )
        self.decoder = nn.Sequential(
            nn.Linear(config['out_channels'], 25),
            nn.LeakyReLU(),
            nn.Linear(25, 50),
            nn.LeakyReLU(),
            nn.Linear(50,75),
            nn.LeakyReLU(), 
            nn.Linear(75, config['window_length'])
        )
    



    def forward(self, x: torch.tensor) -> torch.tensor:
        z = self.encoder(x)

        z = z.squeeze()

        y = self.decoder(z)
        y = y.view((-1,1, self.config['window_length']))

        return y, z
    

    def compute_l1_loss(self, w):
      return torch.abs(w).sum()
  
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    

if __name__ == '__main__':
    Autoencoder('test')