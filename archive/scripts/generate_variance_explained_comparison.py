"""
Professional variance explained comparison: 30D vs 50D latent spaces.
Side-by-side comparison for Chaos, Solitons and Fractals journal.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Professional journal style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
})

from src.models.cvae import LSTM_VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configurations
configs = {
    '30D': {
        'latent_dim': 30,
        'model_path': 'model_ckpts/model_final_30_conditioned.pth',
        'color': '#2E86AB',  # Deep blue
    },
    '50D': {
        'latent_dim': 50,
        'model_path': 'model_ckpts/model_final_50_conditioned.pth',
        'color': '#A23B72',  # Deep magenta
    }
}

TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'


def encode_to_latent_space(latent_dim, model_path):
    """Encode all test samples to latent space."""
    print(f"\nProcessing {latent_dim}D model...")

    # Model config
    model_config = {
        "n_curves": 7,
        "seq_len": 65,
        "latent_dim": latent_dim,
        "rnn_hidden_size": 256,
        "rnn_num_layers": 2,
        "scale_prediction_mode": 'log',
        "use_scale_conditioning": True,
    }

    # Load model
    model = LSTM_VAE(config=model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)
    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    print(f"  Encoding {len(test_data)} samples...")

    n_samples = len(test_data)
    latent_codes = np.zeros((n_samples, latent_dim))

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc=f"  {latent_dim}D encoding", leave=False):
            # Prepare input
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(device)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(device)

            # Encode
            x_rnn = x_norm.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(x_rnn)
            encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

            # Get latent representation
            z_shape = model.shape_projector(encoded_summary)
            z_scale = model.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
            mu = model.fc_mu(bottleneck_input)

            latent_codes[i] = mu.cpu().numpy()[0]

    # PCA
    print(f"  Performing PCA...")
    pca = PCA()
    pca.fit(latent_codes)

    variance_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(variance_ratio)

    # Stats
    n_90 = np.argmax(cumsum >= 0.90) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    n_99 = np.argmax(cumsum >= 0.99) + 1

    print(f"  PCs for 90%: {n_90}, 95%: {n_95}, 99%: {n_99}")

    return variance_ratio, cumsum, n_90, n_95, n_99


def generate_comparison_figure(results_30d, results_50d):
    """Generate professional side-by-side comparison."""
    print("\nGenerating comparison figure...")

    var_30, cum_30, n90_30, n95_30, n99_30 = results_30d
    var_50, cum_50, n90_50, n95_50, n99_50 = results_50d

    color_30 = configs['30D']['color']
    color_50 = configs['50D']['color']

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.7, wspace=0.25,
                           left=0.07, right=0.96, top=0.92, bottom=0.08)

    # ========================================================================
    # Row 1: Full spectrum comparison
    # ========================================================================

    # 30D full spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()

    x_pos_30 = np.arange(len(var_30))
    ax1.bar(x_pos_30, var_30 * 100, alpha=0.7, color=color_30,
            edgecolor='black', linewidth=0.6)
    ax1_twin.plot(x_pos_30, cum_30 * 100, 'o-', color='darkred',
                  linewidth=2, markersize=3, alpha=0.8)

    # Reference lines
    ax1_twin.axhline(90, color='green', linestyle='--', linewidth=1.2, alpha=0.5)
    ax1_twin.axhline(95, color='orange', linestyle='--', linewidth=1.2, alpha=0.5)
    ax1.axvline(n90_30-1, color='green', linestyle=':', linewidth=1.5, alpha=0.4)
    ax1.axvline(n95_30-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.4)

    ax1.set_xlabel('Principal Component', fontweight='bold')
    ax1.set_ylabel('Variance Explained (%)', fontweight='bold', color=color_30)
    ax1_twin.set_ylabel('Cumulative Variance (%)', fontweight='bold', color='darkred')
    ax1.set_title('A) 30D Latent Space - Full Spectrum', fontweight='bold', loc='left', pad=10)

    ax1.tick_params(axis='y', labelcolor=color_30)
    ax1_twin.tick_params(axis='y', labelcolor='darkred')
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_xlim([-0.5, len(var_30) - 0.5])
    ax1_twin.set_ylim([0, 105])

    # Annotations
    ax1.text(n90_30-1, ax1.get_ylim()[1] * 0.9, f'{n90_30}',
             ha='center', va='top', fontsize=9, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

    # 50D full spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()

    x_pos_50 = np.arange(len(var_50))
    ax2.bar(x_pos_50, var_50 * 100, alpha=0.7, color=color_50,
            edgecolor='black', linewidth=0.6)
    ax2_twin.plot(x_pos_50, cum_50 * 100, 'o-', color='darkred',
                  linewidth=2, markersize=3, alpha=0.8)

    # Reference lines
    ax2_twin.axhline(90, color='green', linestyle='--', linewidth=1.2, alpha=0.5)
    ax2_twin.axhline(95, color='orange', linestyle='--', linewidth=1.2, alpha=0.5)
    ax2.axvline(n90_50-1, color='green', linestyle=':', linewidth=1.5, alpha=0.4)
    ax2.axvline(n95_50-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.4)

    ax2.set_xlabel('Principal Component', fontweight='bold')
    ax2.set_ylabel('Variance Explained (%)', fontweight='bold', color=color_50)
    ax2_twin.set_ylabel('Cumulative Variance (%)', fontweight='bold', color='darkred')
    ax2.set_title('B) 50D Latent Space - Full Spectrum', fontweight='bold', loc='left', pad=10)

    ax2.tick_params(axis='y', labelcolor=color_50)
    ax2_twin.tick_params(axis='y', labelcolor='darkred')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_xlim([-0.5, len(var_50) - 0.5])
    ax2_twin.set_ylim([0, 105])

    # Annotations
    ax2.text(n90_50-1, ax2.get_ylim()[1] * 0.9, f'{n90_50}',
             ha='center', va='top', fontsize=9, color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

    # ========================================================================
    # Row 2: Direct comparison
    # ========================================================================

    n_show = 20

    # Overlay comparison of first 20
    ax5 = fig.add_subplot(gs[1, 0])

    width = 0.35
    x_pos = np.arange(n_show)

    ax5.bar(x_pos - width/2, var_30[:n_show] * 100, width,
            label='30D Model', alpha=0.8, color=color_30,
            edgecolor='black', linewidth=0.8)
    ax5.bar(x_pos + width/2, var_50[:n_show] * 100, width,
            label='50D Model', alpha=0.8, color=color_50,
            edgecolor='black', linewidth=0.8)

    ax5.set_xlabel('Principal Component', fontweight='bold')
    ax5.set_ylabel('Variance Explained (%)', fontweight='bold')
    ax5.set_title('C) Direct Comparison - Individual PC Variance', fontweight='bold', loc='left', pad=10)
    ax5.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax5.grid(alpha=0.3, axis='y')
    ax5.set_xlim([-0.5, n_show - 0.5])

    # Cumulative comparison
    ax6 = fig.add_subplot(gs[1, 1])

    ax6.plot(x_pos, cum_30[:n_show] * 100, 'o-', color=color_30,
             linewidth=2.5, markersize=6, label='30D Model', alpha=0.8)
    ax6.plot(x_pos, cum_50[:n_show] * 100, 's-', color=color_50,
             linewidth=2.5, markersize=6, label='50D Model', alpha=0.8)

    # Reference lines
    ax6.axhline(90, color='green', linestyle='--', linewidth=1.5,
                alpha=0.5, label='90% threshold')
    ax6.axhline(95, color='orange', linestyle='--', linewidth=1.5,
                alpha=0.5, label='95% threshold')

    ax6.set_xlabel('Principal Component', fontweight='bold')
    ax6.set_ylabel('Cumulative Variance Explained (%)', fontweight='bold')
    ax6.set_title('D) Direct Comparison - Cumulative Variance', fontweight='bold', loc='left', pad=10)
    ax6.legend(loc='lower right', frameon=True, framealpha=0.95)
    ax6.grid(alpha=0.3)
    ax6.set_xlim([-0.5, n_show - 0.5])
    ax6.set_ylim([0, 105])


    # Save
    output_png = 'final figures/fig_variance_explained_comparison.png'
    output_pdf = 'final figures/fig_variance_explained_comparison.pdf'

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"\nFigure saved:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    plt.close()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\n30D MODEL:")
    print(f"  PCs for 90% variance: {n90_30}")
    print(f"  PCs for 95% variance: {n95_30}")
    print(f"  PCs for 99% variance: {n99_30}")
    print(f"  Top 5 PCs: {cum_30[4]*100:.1f}%")
    print(f"  Top 10 PCs: {cum_30[9]*100:.1f}%")

    print(f"\n50D MODEL:")
    print(f"  PCs for 90% variance: {n90_50}")
    print(f"  PCs for 95% variance: {n95_50}")
    print(f"  PCs for 99% variance: {n99_50}")
    print(f"  Top 5 PCs: {cum_50[4]*100:.1f}%")
    print(f"  Top 10 PCs: {cum_50[9]*100:.1f}%")

    print("\n" + "="*80)


def main():
    print("="*80)
    print("VARIANCE EXPLAINED COMPARISON: 30D vs 50D LATENT SPACES")
    print("="*80)

    # Process both models
    results_30d = encode_to_latent_space(30, configs['30D']['model_path'])
    results_50d = encode_to_latent_space(50, configs['50D']['model_path'])

    # Generate comparison figure
    generate_comparison_figure(results_30d, results_50d)

    print("\n" + "="*80)
    print("✓ Comparison figure generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
