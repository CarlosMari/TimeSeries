import torch
import torch.nn as nn
from torch.nn import functional as F

ACTIVATION = nn.GELU()


class Crop1d(nn.Module):
    def __init__(self, crop_left, crop_right):
        super().__init__()
        self.crop_left = crop_left
        self.crop_right = crop_right

    def forward(self, x):
        if self.crop_left == 0 and self.crop_right == 0:
            return x
        elif self.crop_right == 0:
            return x[:, :, self.crop_left:]
        else:
             # Ensure tensor has enough length before cropping
            if x.size(2) < self.crop_left + self.crop_right:
                 raise ValueError(f"Tensor length {x.size(2)} is smaller than total crop amount {self.crop_left + self.crop_right}")
            return x[:, :, self.crop_left:-self.crop_right]

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

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=config['in_channels'], out_channels=7, kernel_size=5, stride=2, padding=2, groups=1, padding_mode = 'replicate'), # out 65 x 7
            ACTIVATION,
            nn.Conv1d(in_channels=7, out_channels=7, kernel_size=5, stride=2, padding=2, groups=1, padding_mode = 'replicate'), # out 32 x 7
            ACTIVATION,
            nn.Conv1d(in_channels=7, out_channels=10, kernel_size=4, stride=3, padding=2, padding_mode = 'replicate'), # 12 x 10
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
        )
        self.mean_map.apply(init_weights)
        
        self.std_map = torch.nn.Sequential(
            torch.nn.Linear(2*config['latent_dim'], config['latent_dim']),
        )
        self.std_map.apply(init_weights)
        
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], self.flattened_size),
            ACTIVATION,
        )
        self.linear2.apply(init_weights)
        
        # Create dummy encoded data for testing decoder
        with torch.no_grad():
            test_encoded = dummy_output
            B, C, L = test_encoded.shape

            self.decoder = nn.Sequential(
                # Layer 1: Target 33 from input 12 (reverse of Conv3 k=4, s=3)
                nn.Upsample(scale_factor=3, mode='nearest'), # 12 -> 36
                # Conv1d: L_in=36, k=4. Need L_out=33. Formula: L_out = L_in + 2p - k + 1
                # 33 = 36 + 2p - 4 + 1 => 33 = 33 + 2p => p=0.
                nn.Conv1d(in_channels=10, out_channels=7, kernel_size=4, stride=1, padding=0, padding_mode='replicate'), # 36 -> 33
                ACTIVATION,

                # Layer 2: Target 65 from input 33 (reverse of Conv2 k=5, s=2)
                nn.Upsample(scale_factor=2, mode='nearest'), # 33 -> 66
                # Conv1d: L_in=66, k=5. Need L_out=65. Formula: L_out = L_in + 2p - k + 1
                # 65 = 66 + 2p - 5 + 1 => 65 = 62 + 2p => 2p = 3. Not integer.
                # Let's use symmetric padding p=(k-1)//2 = (5-1)//2 = 2
                # L_out = 66 + 2*2 - 5 + 1 = 66 + 4 - 5 + 1 = 66. Need 65.
                nn.Conv1d(in_channels=7, out_channels=7, kernel_size=5, stride=1, padding=2, padding_mode='replicate'), # 66 -> 66
                Crop1d(0, 1), # Crop 1 element from the right: 66 -> 65
                ACTIVATION,

                # Layer 3: Target 129 from input 65 (reverse of Conv1 k=5, s=2)
                nn.Upsample(scale_factor=2, mode='nearest'), # 65 -> 130
                # Conv1d: L_in=130, k=5. Need L_out=129. Formula: L_out = L_in + 2p - k + 1
                # 129 = 130 + 2p - 5 + 1 => 129 = 126 + 2p => 2p = 3. Not integer.
                # Let's use symmetric padding p=(k-1)//2 = (5-1)//2 = 2
                # L_out = 130 + 2*2 - 5 + 1 = 130 + 4 - 5 + 1 = 130. Need 129.
                nn.Conv1d(in_channels=7, out_channels=config["in_channels"], kernel_size=5, stride=1, padding=2, padding_mode='replicate'), # 130 -> 130
                Crop1d(0, 1), # Crop 1 element from the right: 130 -> 129
                # No final activation usually needed if output is e.g. raw signal/image
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
