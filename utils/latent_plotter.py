import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import sys

sys.path.insert(1, "/Users/carlosmarinoguera/Documents/IIT/OriginalPaper")
print(sys.path[0])
from VAE.models.CH_VAE import CHVAE
from config import model_config

def visualize_vae_multiple_samples(model, X, n_curves=7, n_samples=100, alpha=0.1, device='mps'):
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

    indices = np.arange(n_curves)
    subsets = X[23].to(device)

    originals = [subset.cpu().numpy() for subset in subsets]

    # Generate multiple reconstructions for each curve
    reconstructions = []
    latent_codes = []
    
    # Colors for different curves
    colors = plt.cm.tab20(np.linspace(0, 1, n_curves))
    
    # Plot original curves
    plt.figure(figsize=(15, 8))
    for i, (orig, color) in enumerate(zip(originals, colors)):
        plt.plot(orig, color=color, linewidth=2, label=f'Original Curve {i+1}')
    plt.title('Original Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.show()
    
    # Reconstruction phase
    model.eval()
    torch_subsets = torch.tensor(subsets).unsqueeze(0)
    reconstruct = []
    codes =  []
    for i in range(n_samples):
        with torch.no_grad():
            recons,  cod, *_ = model(torch_subsets)
            reconstruct.append(recons.squeeze(0))
            codes.append(cod)

    
    print(f'{len(codes)=}, {codes[0].shape}')
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
    
    # Latent Space Visualization
    # 1. Latent Space Histograms
    print(f'{latent_codes.shape}')
    n_latent_dims = latent_codes[0].shape[1]
    
    # Select first 10 dimensions for histogram (or all if less than 10)
    dims_to_plot = min(10, n_latent_dims)
    fig, axs = plt.subplots(2, (dims_to_plot + 1) // 2, figsize=(20, 8))
    axs = axs.ravel()
    
    for dim in range(dims_to_plot):
        for i, (codes, color) in enumerate(zip(latent_codes, colors)):
            # Histogram
            axs[dim].hist(codes[:, dim], bins=30, density=True, alpha=0.5, 
                          color=color, label=f'Curve {i+1}')
            
            # Fit normal distribution
            mu, std = stats.norm.fit(codes[:, dim])
            xmin, xmax = axs[dim].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            axs[dim].plot(x, p, '--', color=color, linewidth=2)
            axs[dim].text(0.02, 0.98 - i*0.1, f'Curve {i+1}: μ={mu:.2f}, σ={std:.2f}',
                          transform=axs[dim].transAxes, color=color)
        
        axs[dim].set_title(f'Latent Dimension {dim+1}')
        axs[dim].grid(True)
        axs[dim].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 2. Dimensionality Reduction Techniques for Latent Space
    # Combine all latent codes
    all_latent_codes = np.concatenate(latent_codes)
    all_labels = np.concatenate([np.full(codes.shape[0], i) for i, codes in enumerate(latent_codes)])
    
    # PCA Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_latent_codes)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    for i, color in enumerate(colors):
        mask = all_labels == i
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                    color=color, label=f'Curve {i+1}', alpha=0.7)
    plt.title('Latent Space - PCA')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    
    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_latent_codes)
    
    plt.subplot(122)
    for i, color in enumerate(colors):
        mask = all_labels == i
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                    color=color, label=f'Curve {i+1}', alpha=0.7)
    plt.title('Latent Space - t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # UMAP Visualization (optional, requires umap-learn)
    try:
        import umap
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_result = umap_reducer.fit_transform(all_latent_codes)
        
        plt.figure(figsize=(8, 6))
        for i, color in enumerate(colors):
            mask = all_labels == i
            plt.scatter(umap_result[mask, 0], umap_result[mask, 1], 
                        color=color, label=f'Curve {i+1}', alpha=0.7)
        plt.title('Latent Space - UMAP')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.show()
    except ImportError:
        print("UMAP visualization skipped. Install umap-learn for this visualization.")



if __name__ == '__main__':
    # Your data loading code
    DEVICE = 'mps'
    with open('data/TRAIN.pkl', 'rb') as file:
        X = pickle.load(file)

    file.close()
    X = (X - X.min())/(X.max() - X.min())
    X = torch.from_numpy(X).to(torch.float32)
    # Load model
    model = CHVAE(model_config).to(DEVICE)
    model.load_state_dict(torch.load('model_ckpts/model.pth'))

    # Run visualization with desired number of curves
    n_curves = 7  # Change this to compare different number of curves
    visualize_vae_multiple_samples(
        model, X, n_curves=n_curves, n_samples=1000, alpha=0.1
    )