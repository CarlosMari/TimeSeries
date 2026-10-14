import torch

def alpha_div_loss(x_hat, x, mu, log_var, z, a_weight, alpha=1, len_dataset=10, beta=1):
        epsilon = 1e-5
        MSE = torch.sum((x_hat - x) ** 2, dim=-1)
        exp_mse = torch.exp(-alpha * 0.5 * MSE)
        s_i = torch.mean(exp_mse, dim=1)
        L_i = (1 / alpha) * torch.log(s_i + 1e-10)

        recon_loss = -torch.sum(L_i)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = 2 * (len_dataset / x.size(0)) * recon_loss + beta  * kl_divergence

        return total_loss, (len_dataset / x.size(0)) * recon_loss, kl_divergence