import torch
import torch.nn as nn
from torch.nn import functional as F

ACTIVATION = nn.GELU()

class CHVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        def init_weights(m):
            pass
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                # Initialize weights normally
                nn.init.xavier_normal_(m.weight)
                # Initialize bias with higher variance
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)


        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=config['in_channels'], out_channels=16, kernel_size=7, stride=2, padding=3),
            ACTIVATION,
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2),
            ACTIVATION,
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            ACTIVATION,
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1),
            ACTIVATION,
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            ACTIVATION,
        )

        self.encoder.apply(init_weights)

        self.corr = torch.nn.Sequential(
            torch.nn.Linear(36*2, 2*config['latent_dim']),
            ACTIVATION,
        )

        self.corr.apply(init_weights)

        self.mean_map = torch.nn.Sequential(
            torch.nn.Linear(2*config['latent_dim'], 40),
            ACTIVATION,
            torch.nn.Linear(40, 40),
            ACTIVATION,
            torch.nn.Linear(40, config["latent_dim"])
        )

        self.mean_map.apply(init_weights)

        self.std_map = torch.nn.Sequential(
            torch.nn.Linear(2*config['latent_dim'], 40),
            ACTIVATION,
            torch.nn.Linear(40, 40),
            ACTIVATION,
            torch.nn.Linear(40, config["latent_dim"])
        )

        self.std_map.apply(init_weights)

        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], config["latent_dim"]),
            ACTIVATION,
            torch.nn.Linear(config["latent_dim"], config["latent_dim"]*2),
            ACTIVATION,
            torch.nn.Linear(config["latent_dim"]*2, 36*2),
            ACTIVATION,
        )

        self.linear2.apply(init_weights)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 8, kernel_size = 3, padding=1),
            ACTIVATION,
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 8, kernel_size = 3, stride=2, padding=1),
            ACTIVATION,
            torch.nn.ConvTranspose1d(in_channels = 8, out_channels = 16, kernel_size = 5, stride=2, padding=2),
            ACTIVATION,
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = 16,  kernel_size = 5, stride=2, padding=2),
            ACTIVATION,
            torch.nn.ConvTranspose1d(in_channels = 16, out_channels = config["in_channels"], kernel_size=7, stride=2, padding=3),
        )

        self.decoder.apply(init_weights)

    def sample(self, mean, log_var):
        """Sample a given N(0,1) normal distribution given a mean and log of variance."""
        
        # First compute the variance from the log variance. 
        var = torch.clamp(torch.exp(0.5*log_var),min=1e-4)
        
        # Compute a scaled distribution
        eps = torch.randn_like(var)
        
        # Add the vectors
        z = mean + var*eps
        
        return z
    

    def forward(self, X):
        """Forward propogate through the model, return both the reconstruction and sampled mean and standard deviation
        for the system. 
        """
        
        shape = X.shape
        #X_0 = X[:,:,0]
        # Now pass the information through the convolutional feature extracto
        pre_code = self.encoder(X)
        # Reshape the tensor dimension for latent space sampling
        
        B, C, L = pre_code.shape[0], pre_code.shape[1], pre_code.shape[2]
        flattened = pre_code.view(B,C*L)
        flattened = self.corr(flattened)        
        # Now sample from the latent distribution for these points
        mu = self.mean_map(flattened)
        log_var = self.std_map(flattened)
        
        code = self.sample(mu, log_var)
        
        post_code = self.linear2(code)

        X_hat = self.decoder( post_code.view(B,C,L))

        X_hat = X_hat.view(shape)

        return X_hat, code, mu, log_var
    

    @staticmethod
    def loss(x_hat, x, mu, log_var, z, a_weight, alpha = 0.5):
        "Compute the sum of BCE and KL loss for the distribution."
        # x_hat -> BATCH, 7, 129
        #MSE = F.mse_loss(x_hat, x)
        
        MSE = torch.sum((x_hat - x) ** 2, dim=-1) # (BATCH_SIZE, 7)
        exp_mse = torch.exp(-alpha * 0.5 * MSE)
        s_i = torch.mean(exp_mse, dim = 1) # (BATCH_SIZE)
        L_i = (1/alpha) * torch.log(s_i + 1e-10)

        recon_loss = -torch.sum(L_i)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = recon_loss + 0.01 * kl_divergence

        return total_loss, recon_loss, kl_divergence