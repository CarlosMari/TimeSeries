"""
Generate variance explained figure showing importance of each latent dimension.

This script:
1. Loads the scale-conditioned CVAE model
2. Encodes all test samples to latent space
3. Performs PCA on latent codes
4. Visualizes variance explained by each principal component
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.0,
})

# Import model and config
from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE

# Model configuration
model_config = {
    "in_channels": 7,
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 50,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

# Paths
TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
MODEL_PATH = 'model_ckpts/model_final_50_conditioned.pth'
OUTPUT_DIR = 'final figures/'


def load_model_and_data():
    """Load the trained model and test dataset."""
    print("Loading model and data...")

    # Load model
    model = LSTM_VAE(config=model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load test data
    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)

    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    print(f"Model loaded. Test samples: {len(test_data)}")
    return model, test_data, test_max_vals


def encode_to_latent_space(model, test_data, test_max_vals):
    """
    Encode all test samples to latent space.

    Returns:
        latent_codes (numpy array): Shape (N, latent_dim) - the mu values from encoder
    """
    print("Encoding all test samples to latent space...")

    n_samples = len(test_data)
    latent_dim = model_config['latent_dim']
    latent_codes = np.zeros((n_samples, latent_dim))

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Encoding samples"):
            # Prepare input
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Encode
            x_rnn = x_norm.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(x_rnn)
            encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

            # Get latent representation (conditioning path)
            z_shape = model.shape_projector(encoded_summary)
            z_scale = model.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)

            # Get mu (the mean of the latent distribution)
            mu = model.fc_mu(bottleneck_input)

            latent_codes[i] = mu.cpu().numpy()[0]

    print(f"Encoded {n_samples} samples to latent space (dim={latent_dim})")
    return latent_codes


def generate_variance_explained_figure(latent_codes):
    """
    Generate comprehensive variance explained visualization.

    Creates a figure with:
    - Bar plot of variance explained by each PC
    - Cumulative variance explained line
    - Reference lines at 90%, 95%, 99%
    """
    print("\nPerforming PCA analysis...")

    # Fit PCA
    pca = PCA()
    pca.fit(latent_codes)

    variance_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(variance_ratio)

    # Find how many components explain certain thresholds
    n_90 = np.argmax(cumsum >= 0.90) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    n_99 = np.argmax(cumsum >= 0.99) + 1

    print(f"\nVariance explained statistics:")
    print(f"  Components for 90% variance: {n_90}/{len(variance_ratio)}")
    print(f"  Components for 95% variance: {n_95}/{len(variance_ratio)}")
    print(f"  Components for 99% variance: {n_99}/{len(variance_ratio)}")
    print(f"  Total variance in first 10 PCs: {cumsum[9]:.2%}")
    print(f"  Total variance in first 20 PCs: {cumsum[19]:.2%}")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # ========== Panel 1: Full spectrum ==========
    n_dims = len(variance_ratio)
    x_pos = np.arange(n_dims)

    # Bar plot
    bars = ax1.bar(x_pos, variance_ratio * 100,
                   alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8,
                   label='Individual PC')

    # Cumulative line
    ax1_twin = ax1.twinx()
    line = ax1_twin.plot(x_pos, cumsum * 100, 'ro-', linewidth=2.5,
                         markersize=4, label='Cumulative', alpha=0.8)

    # Reference lines
    ax1_twin.axhline(y=90, color='green', linestyle='--', linewidth=1.5,
                     alpha=0.6, label='90% threshold')
    ax1_twin.axhline(y=95, color='orange', linestyle='--', linewidth=1.5,
                     alpha=0.6, label='95% threshold')
    ax1_twin.axhline(y=99, color='red', linestyle='--', linewidth=1.5,
                     alpha=0.6, label='99% threshold')

    # Mark threshold components
    ax1.axvline(x=n_90-1, color='green', linestyle=':', linewidth=1.5, alpha=0.4)
    ax1.axvline(x=n_95-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.4)
    ax1.axvline(x=n_99-1, color='red', linestyle=':', linewidth=1.5, alpha=0.4)

    ax1.set_xlabel('Principal Component Index', fontweight='bold')
    ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
    ax1_twin.set_ylabel('Cumulative Variance Explained (%)', fontweight='bold', color='red')
    ax1.set_title(f'Latent Space Structure: PCA Variance Explained (50D Latent Space)',
                  fontweight='bold', fontsize=13)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='right', framealpha=0.95)

    ax1.grid(alpha=0.3, axis='y')
    ax1.set_xlim([-0.5, n_dims - 0.5])
    ax1.set_ylim([0, max(variance_ratio * 100) * 1.1])
    ax1_twin.set_ylim([0, 105])
    ax1_twin.tick_params(axis='y', labelcolor='red')

    # Add text annotations
    ax1.text(n_90-1, ax1.get_ylim()[1] * 0.95, f'PC{n_90}\n90%',
             ha='center', va='top', fontsize=9, color='green', fontweight='bold')
    ax1.text(n_95-1, ax1.get_ylim()[1] * 0.85, f'PC{n_95}\n95%',
             ha='center', va='top', fontsize=9, color='orange', fontweight='bold')

    # ========== Panel 2: Zoomed view of first 20 components ==========
    n_show = min(20, n_dims)
    x_pos_zoom = np.arange(n_show)

    # Bar plot
    bars2 = ax2.bar(x_pos_zoom, variance_ratio[:n_show] * 100,
                    alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8,
                    label='Individual PC')

    # Cumulative line
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(x_pos_zoom, cumsum[:n_show] * 100, 'ro-',
                          linewidth=2.5, markersize=6, label='Cumulative', alpha=0.8)

    # Highlight top 5
    for i in range(min(5, n_show)):
        bars2[i].set_color('darkblue')
        bars2[i].set_alpha(0.9)

    ax2.set_xlabel('Principal Component Index', fontweight='bold')
    ax2.set_ylabel('Variance Explained (%)', fontweight='bold')
    ax2_twin.set_ylabel('Cumulative Variance Explained (%)', fontweight='bold', color='red')
    ax2.set_title(f'First {n_show} Principal Components (Detail View)',
                  fontweight='bold', fontsize=13)

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars2, variance_ratio[:n_show] * 100)):
        height = bar.get_height()
        if i < 10:  # Only label first 10 to avoid clutter
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=8, rotation=0)

    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='right', framealpha=0.95)

    ax2.grid(alpha=0.3, axis='y')
    ax2.set_xlim([-0.5, n_show - 0.5])
    ax2_twin.set_ylim([0, 105])
    ax2_twin.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()

    # Save figure
    output_path_png = f'{OUTPUT_DIR}fig_variance_explained.png'
    output_path_pdf = f'{OUTPUT_DIR}fig_variance_explained.pdf'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path_png}")
    print(f"Saved: {output_path_pdf}")
    plt.close()

    return variance_ratio, cumsum


def main():
    """Main execution function."""
    print("="*80)
    print("GENERATING VARIANCE EXPLAINED FIGURE")
    print("Scale-Conditioned CVAE - Latent Space Analysis")
    print("="*80)

    # Load model and data
    model, test_data, test_max_vals = load_model_and_data()

    # Encode to latent space
    latent_codes = encode_to_latent_space(model, test_data, test_max_vals)

    # Generate figure
    variance_ratio, cumsum = generate_variance_explained_figure(latent_codes)

    print("\n" + "="*80)
    print("VARIANCE EXPLAINED FIGURE GENERATED SUCCESSFULLY!")
    print(f"Output: {OUTPUT_DIR}fig_variance_explained.png/pdf")
    print("="*80)


if __name__ == "__main__":
    main()
