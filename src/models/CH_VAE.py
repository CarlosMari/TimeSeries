import torch
import torch.nn as nn

ACTIVATION = nn.SiLU()


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
            if x.size(2) < self.crop_left + self.crop_right:
                raise ValueError(f"Tensor length {x.size(2)} is smaller than total crop amount {self.crop_left + self.crop_right}")
            return x[:, :, self.crop_left:-self.crop_right]


class CHVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.bias, mean=0.0, std=0.05)
                nn.init.normal_(m.weight, mean=0.0, std=1)

        self.encoder = nn.Sequential(
            nn.Conv1d(config['in_channels'], 7, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            ACTIVATION,
            nn.Conv1d(7, 7, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            ACTIVATION,
            nn.Conv1d(7, 10, kernel_size=4, stride=3, padding=2, padding_mode='replicate'),
            ACTIVATION,
        )

        dummy_input = torch.zeros(1, config['in_channels'], 129)
        dummy_output = self.encoder(dummy_input)
        self.flattened_size = dummy_output.size(1) * dummy_output.size(2)
        self.encoder_output_channels = dummy_output.size(1)
        self.encoder_output_length = dummy_output.size(2)

        self.corr = nn.Sequential(
            nn.Linear(self.flattened_size, 2 * config['latent_dim']),
            ACTIVATION,
        )
        self.corr.apply(init_weights)

        self.mean_map = nn.Sequential(
            nn.Linear(2 * config['latent_dim'], config['latent_dim']),
        )
        self.mean_map.apply(init_weights)

        self.std_map = nn.Sequential(
            nn.Linear(2 * config['latent_dim'], config['latent_dim']),
        )
        self.std_map.apply(init_weights)

        self.linear2 = nn.Sequential(
            nn.Linear(config["latent_dim"], self.flattened_size),
            ACTIVATION,
        )
        self.linear2.apply(init_weights)

        with torch.no_grad():
            B, C, L = dummy_output.shape
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=3, mode='nearest'),
                nn.Conv1d(10, 7, kernel_size=4, stride=1, padding=0, padding_mode='replicate'),
                ACTIVATION,
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(7, 7, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
                Crop1d(0, 1),
                ACTIVATION,
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(7, config["in_channels"], kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
                Crop1d(0, 1),
            )

            test_output = self.decoder(torch.zeros(1, C, L))
            if test_output.shape[2] != 129:
                raise ValueError(f"Decoder output length ({test_output.shape[2]}) does not match required 129")

    def sample(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        eps = torch.randn_like(var)
        return mean + var * eps

    def forward(self, X):
        original_shape = X.shape

        pre_code = self.encoder(X)
        B, C, L = pre_code.shape
        flattened = pre_code.view(B, C * L)
        flattened = self.corr(flattened)

        mu = self.mean_map(flattened)
        log_var = self.std_map(flattened)
        code = self.sample(mu, log_var)

        post_code = self.linear2(code)
        reshaped = post_code.view(B, C, L)
        X_hat = self.decoder(reshaped)
        X_hat = torch.clamp(X_hat, min=0)

        if X_hat.shape != original_shape:
            print(f"Warning: Output shape {X_hat.shape} doesn't match input shape {original_shape}")

        return X_hat, code, mu, log_var

    @staticmethod
    def loss(x_hat, x, mu, log_var, z, a_weight, alpha=1, len_dataset=10, beta=1):
        epsilon = 1e-5
        MSE = torch.sum((x_hat - x) ** 2, dim=-1)
        exp_mse = torch.exp(-alpha * 0.5 * MSE)
        s_i = torch.mean(exp_mse, dim=1)
        L_i = (1 / alpha) * torch.log(s_i + 1e-10)

        recon_loss = -torch.sum(L_i)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = 2 * (len_dataset / x.size(0)) * recon_loss + beta  * kl_divergence

        return total_loss, (len_dataset / x.size(0)) * recon_loss, kl_divergence
