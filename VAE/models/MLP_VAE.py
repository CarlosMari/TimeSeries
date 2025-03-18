import torch
import torch.nn as nn
from torch.nn import functional as F

ACTIVATION = nn.GELU()

class MLPVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config


        # Encoder remains the same as before
        self.encoder = nn.Sequential(
            nn.Linear(7*129, 3*129),
            ACTIVATION,
            nn.Linear(3*129, 129),
            ACTIVATION,
            nn.Linear(129, 40),
            ACTIVATION,
        )
        
        self.mean_map = torch.nn.Sequential(
            torch.nn.Linear(40, config["latent_dim"]),
        )
        
        self.std_map = torch.nn.Sequential(
            torch.nn.Linear(40, config["latent_dim"])
        )
        
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], 40),
            ACTIVATION,
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(40, 129),
            ACTIVATION,
            nn.Linear(129, 3*129),
            ACTIVATION,
            nn.Linear(3*129, 7*129),
        )
        

    def sample(self, mean, log_var):
        """Sample a given N(0,1) normal distribution given a mean and log of variance."""
        var = torch.exp(0.5*log_var)
        eps = torch.randn_like(var)
        z = mean + var*eps
        return z
    
    def forward(self, X):
        """Forward propagate through the model"""
        original_shape = X.shape
        
        pre_code = self.encoder(X.reshape(-1, 7*129))
        
        # Sample from latent distribution
        mu = self.mean_map(pre_code)
        log_var = self.std_map(pre_code)
        code = self.sample(mu, log_var)
        
        # Decode
        post_code = self.linear2(code)
        X_hat = self.decoder(post_code)
        X_hat = X_hat.reshape(-1, 7, 129)

        if X_hat.shape != original_shape:
            print(f"Warning: Output shape {X_hat.shape} doesn't match input shape {original_shape}")
        
        return X_hat, code, mu, log_var
    
    @staticmethod
    def loss(x_hat, x, mu, log_var, z, a_weight, alpha=0.5, len_dataset=10, beta=1):
        """Compute the sum of BCE and KL loss for the distribution."""
        # Calculate MSE


        #print(f'{x_hat.shape= } \n {x.shape=}')
        MSE = torch.sum((x_hat - x) ** 2, dim=-1)  # (BATCH_SIZE, 7)

        batch_size = MSE.shape[0]
        exp_mse = torch.exp(-alpha * 0.5 * MSE)
        s_i = torch.mean(exp_mse, dim=1)  # (BATCH_SIZE)
        L_i = (1/alpha) * torch.log(s_i + 1e-10)

        recon_loss = -torch.sum(L_i)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        beta = 0
        total_loss = 2*(len_dataset/batch_size) * recon_loss + beta * kl_divergence

        return total_loss, (len_dataset/batch_size) * recon_loss, kl_divergence