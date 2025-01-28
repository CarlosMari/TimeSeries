import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from scipy import stats
import sys

from VAE.models.CH_VAE import CHVAE
from config import model_config

def visualize_vae_multiple_samples(model, X, n_curves=7, n_samples=100, alpha=0.1, device='cuda', iterations=10):
    """
    Visualize VAE reconstructions and latent space for multivariate time series.
    
    Parameters:
    - model: PyTorch VAE model
    - X: Input tensor of shape (batch_size, 7, 129)
    - n_curves: Number of curves to visualize (default 7)
    - n_samples: Number of reconstructions per curve
    - alpha: Transparency of reconstruction lines
    - device: Device to run computations on
    """
    # Ensure model and data are on the correct device
    model = model.to(device)
    
    # Select curves (in this case, all 7 curves)
    if n_curves > X.shape[0]:
        n_curves = X.shape[0]
    assert n_curves == 7, f'{n_curves=}'

    
    for it in range(iterations):
        subsets = X[it].to(device)

        originals = [subset.cpu().numpy() for subset in subsets]

        # Generate multiple reconstructions for each curve
        reconstructions = []
        latent_codes = []
        
        # Colors for different curves
        colors = plt.cm.tab20(np.linspace(0, 1, n_curves)) 
        
        # Reconstruction phase
        model.eval()
        torch_subsets = subsets.clone().detach().unsqueeze(0)
        reconstruct = []
        codes =  []
        for i in range(n_samples):
            with torch.no_grad():
                recons,  cod, *_ = model(torch_subsets)
                reconstruct.append(recons.squeeze(0))
                codes.append(cod)

        reconstruct = torch.stack(reconstruct, dim=0)
        codes = torch.stack(codes, dim = 0)
        reconstructions = reconstruct.permute(1, 0, 2).cpu().numpy()
        latent_codes = codes.permute(1, 0, 2).cpu().numpy()
        
        # Reconstruction visualization
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        
        for i, (recons, orig, color) in enumerate(zip(reconstructions, originals, colors)):

            axis = int(i / 3), i%3
            for z in range(n_samples):
                axs[axis].plot(recons[z], '-', color=color, alpha=alpha, linewidth=0.2)
            
            # Plot original
            axs[axis].plot(orig, 'r-', linewidth=2, 
                        label=f'Original Curve {i+1}')
                
            # Add mean reconstruction
            mean_recon = recons.mean(axis=0)
            axs[axis].plot(mean_recon, '--', color='black', linewidth=2, 
                        label='Mean Reconstruction')
        
            axs[axis].set_title(f'{n_samples} Reconstructions for Curve {i+1}')
            axs[int(i / 3), i%3].grid(True)
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'./generation_comparison/test_{it}.png')
    

if __name__ == '__main__':
    # Your data loading code
    DEVICE = 'cuda'
    with open('data/TEST_NEW_DIST.pkl', 'rb') as file:
        X = pickle.load(file)

    file.close()
    X = (X - X.min())/(X.max() - X.min())
    X = torch.from_numpy(X).to(torch.float32)
    # Load model
    model = CHVAE(model_config).to(DEVICE)
    model.load_state_dict(torch.load('model_ckpts/model_alpha_weight.pth', map_location=torch.device('cpu')))

    # Run visualization with desired number of curves
    n_curves = 7  # Change this to compare different number of curves
    visualize_vae_multiple_samples(
        model, X, n_curves=n_curves, n_samples=1000, alpha=0.1
    )