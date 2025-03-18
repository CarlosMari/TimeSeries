import torch
import torch.nn as nn
from torch.nn import functional as F

ACTIVATION = nn.GELU()

class CHVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                # Initialize weights normally
                nn.init.xavier_normal_(m.weight)
                # Initialize bias with higher variance
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)

        # Encoder remains the same as before
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=config['in_channels'], out_channels=7, kernel_size=5, stride=1, padding=2), # out 45 x 10
            ACTIVATION,
            nn.Conv1d(in_channels=7, out_channels=10, kernel_size=5, stride=1, padding=2), # out 22 x 15
            ACTIVATION,
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=4, stride=1, padding=2), # 11 x 15
            ACTIVATION,
        )
        self.encoder.apply(init_weights)
        
        # Calculate the actual flattened size after encoder
        dummy_input = torch.zeros(1, config['in_channels'], 129)
        dummy_output = self.encoder(dummy_input)
        print(f'{dummy_output.shape=}')
        self.flattened_size = dummy_output.size(1) * dummy_output.size(2)
        
        # Store encoder dimensions for decoder reshaping
        self.encoder_output_channels = dummy_output.size(1)
        self.encoder_output_length = dummy_output.size(2)
        print(f"Encoder output shape: {dummy_output.shape} (flattened: {self.flattened_size})")
        
        # Rest of the network unchanged
        self.corr = torch.nn.Sequential(
            torch.nn.Linear(self.flattened_size, 2*config['latent_dim']),
            ACTIVATION,
        )
        self.corr.apply(init_weights)
        
        self.mean_map = torch.nn.Sequential(
            torch.nn.Linear(2*config['latent_dim'], config['latent_dim']),
            #ACTIVATION,
            #torch.nn.Linear(15, config["latent_dim"])
        )
        self.mean_map.apply(init_weights)
        
        self.std_map = torch.nn.Sequential(
            torch.nn.Linear(2*config['latent_dim'], config['latent_dim']),
            #ACTIVATION,
            #torch.nn.Linear(15, config["latent_dim"])
        )
        self.std_map.apply(init_weights)
        
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], self.flattened_size),
            ACTIVATION,
            #torch.nn.Linear(config["latent_dim"], self.flattened_size),
            #ACTIVATION,
        )
        self.linear2.apply(init_weights)

        # We need to precisely calculate the padding and output_padding for each layer
        # to ensure our final output has exactly 129 length
        
        # Create dummy encoded data for testing decoder
        with torch.no_grad():
            test_encoded = dummy_output
            B, C, L = test_encoded.shape

            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(
                    in_channels=10, 
                    out_channels=10, 
                    kernel_size=4,
                    stride=1,
                    padding=2, 
                    output_padding=0
                ),
                ACTIVATION,
                torch.nn.ConvTranspose1d(
                    in_channels=10, 
                    out_channels=7, 
                    kernel_size=5,
                    stride=1,
                    padding=2,  # Adjusted padding
                    output_padding=0  # Added output padding
                ),
                ACTIVATION,
                torch.nn.ConvTranspose1d(
                    in_channels=7, 
                    out_channels=config["in_channels"], 
                    kernel_size=5,
                    stride=1,
                    padding=2,  # Adjusted padding
                    output_padding=0  # Added output padding
                ),
            )
            self.decoder.apply(init_weights)
            
            # Verify the dimensions
            test_input = torch.zeros(1, self.encoder_output_channels, self.encoder_output_length)
            print(f'{test_input.shape=}')
            test_output = self.decoder(test_input)
            print(f"Decoder output shape: {test_output.shape}")
            
            if test_output.shape[2] != 129:
                raise ValueError(f"Decoder output length ({test_output.shape[2]}) does not match required 129")

    def sample(self, mean, log_var):
        """Sample a given N(0,1) normal distribution given a mean and log of variance."""
        var = torch.exp(0.5*log_var)
        eps = torch.randn_like(var)
        z = mean + var*eps
        return z
    
    def forward(self, X):
        """Forward propagate through the model"""
        original_shape = X.shape
        
        # Encode
        pre_code = self.encoder(X)
        
        # Reshape for latent space
        B, C, L = pre_code.shape
        flattened = pre_code.view(B, C*L)
        flattened = self.corr(flattened)
        
        # Sample from latent distribution
        mu = self.mean_map(flattened)
        log_var = self.std_map(flattened)
        code = self.sample(mu, log_var)
        
        # Decode
        post_code = self.linear2(code)
        reshaped = post_code.view(B, C, L)
        X_hat = self.decoder(reshaped)
        
        # Final verification of output dimensions
        if X_hat.shape != original_shape:
            print(f"Warning: Output shape {X_hat.shape} doesn't match input shape {original_shape}")
        
        return X_hat, code, mu, log_var
    
    @staticmethod
    def loss(x_hat, x, mu, log_var, z, a_weight, alpha=0.5, len_dataset=10, beta=1):
        """Compute the sum of BCE and KL loss for the distribution."""
        # Calculate MSE
        MSE = torch.sum((x_hat - x) ** 2, dim=-1)  # (BATCH_SIZE, 7)

        batch_size = MSE.shape[0]
        exp_mse = torch.exp(-alpha * 0.5 * MSE)
        s_i = torch.mean(exp_mse, dim=1)  # (BATCH_SIZE)
        L_i = (1/alpha) * torch.log(s_i + 1e-10)

        recon_loss = -torch.sum(L_i)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = 2*(len_dataset/batch_size) * recon_loss #+ beta * 0.1 * kl_divergence

        return total_loss, (len_dataset/batch_size) * recon_loss, kl_divergence