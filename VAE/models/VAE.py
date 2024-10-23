import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
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
            nn.Conv1d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        # Mean mapping

        self.mean_map = torch.nn.Sequential(
            torch.nn.Linear(18, 24),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(24, 24),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(24, config["latent_dim"])
        )

        self.std_map = torch.nn.Sequential(
            torch.nn.Linear(18, 24),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(24, 24),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(24, config["latent_dim"])
        )

        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], 24),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(24, 24),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(24, 18),
            torch.nn.LeakyReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels = 2, out_channels = 8, kernel_size = 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 8, kernel_size = 3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 16, kernel_size = 5, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = 16,  kernel_size = 5, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = config["in_channels"], kernel_size=7, stride=2, padding=3),
        )


    def sample(self, mean, log_var):
        """Sample a given N(0,1) normal distribution given a mean and log of variance."""
        
        # First compute the variance from the log variance. 
        var = torch.exp(0.5*log_var)
        
        # Compute a scaled distribution
        eps = torch.randn_like(var)
        
        # Add the vectors
        z = mean + var*eps
        
        return z
    

    def forward(self, X):
        """Forward propogate through the model, return both the reconstruction and sampled mean and standard deviation
        for the system. 
        """
        
        # Now pass the information through the convolutional feature extracto
        pre_code = self.encoder(X)
                
        # Get the dimensionf othe precode
       
        
        # Reshape the tensor dimension for latent space sampling
        

        B, C, L = pre_code.shape[0], pre_code.shape[1], pre_code.shape[2]
        flattened = pre_code.view(B,C*L)
        
        # Now sample from the latent distribution for these points
        mu = self.mean_map(flattened)
        log_var = self.std_map(flattened)
        
        code = self.sample(mu, log_var)
        
        # Now pass the information through the decoder. Note we pass the last layer through a sigmoid
        # for the BCS loss
        post_code = self.linear2(code)
        
        
        X_hat = self.decoder( post_code.view(B, C, L)).squeeze()
                          
        #print(f'Getting X_hat shape: {X_hat.shape}')
        return X_hat, code, mu, log_var
    

    @staticmethod
    def loss(x_hat, x, mu, log_var, alpha, gamma = 0):
        "Compute the sum of BCE and KL loss for the distribution."

        # Compute the reconstruction loss
        x_hat = x_hat.unsqueeze(1)
        #print(f"x_hat: {x_hat.shape}, x: {x.shape}")
        BCE = F.mse_loss(x_hat, x)

        # Compute the KL divergence of the distribution. 
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Normalize by the number of points in the batch and the dimensionality
        KLD /= (x.shape[0]*x.shape[1])
        
        SSL = F.mse_loss(x_hat[:,:,-1], x[:,:,-1])
        # ICL = F.mse_loss(x_hat[:,:,0], x[:,:,0])

        return BCE + alpha*KLD + gamma*SSL