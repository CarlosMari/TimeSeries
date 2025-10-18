import torch
import torch.nn as nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config


        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=config['in_channels'], out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        # Mean mapping

        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(18*4, config["latent_dim"]*12),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(config["latent_dim"]*12, config["latent_dim"]*12),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(config["latent_dim"]*12, config["latent_dim"])
        )

        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], config["latent_dim"]*12),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(config["latent_dim"]*12, config["latent_dim"]*12),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(config["latent_dim"]*12, 18*4),
            torch.nn.LeakyReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 8, kernel_size = 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 8, kernel_size = 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 16, kernel_size = 5, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = 16,  kernel_size = 5, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = config["in_channels"], kernel_size=7, stride=2, padding=3),
        )

    

    def forward(self, X):
        """Forward propogate through the model, return both the reconstruction and sampled mean and standard deviation
        for the system. 
        """
        pre_code = self.encoder(X)
        B, C, L = pre_code.shape[0], pre_code.shape[1], pre_code.shape[2]
        flattened = pre_code.view(B,C*L)
        code = self.linear1(flattened)
        
        post_code = self.linear2(code)
        
        X_hat = self.decoder( post_code.view(B, C, L)).squeeze()
                          
        return X_hat, code, -1, -1
    

    @staticmethod
    def loss(x_hat, x, *args):
        "Compute the sum of BCE and KL loss for the distribution."
        BCE = F.mse_loss(x_hat, x)

        return BCE 