"""
Generate comprehensive phase space comparison: Real vs Generated trajectories.

This script creates detailed phase space visualizations comparing real and generated
Lotka-Volterra dynamics in the same plots to show overlap and quality.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Publication-quality plot settings
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
    """Load trained model and test dataset."""
    print("Loading model and data...")

    model = LSTM_VAE(config=model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)

    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    print(f"Model loaded. Test samples: {len(test_data)}")
    return model, test_data, test_max_vals


def generate_samples(model, n_samples=100):
    """Generate samples from prior."""
    print(f"Generating {n_samples} samples from prior...")

    generated_data = []
    predicted_max_vals = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Generating"):
            z = torch.randn(1, model_config['latent_dim']).to(DEVICE)
            recon_norm, max_vals_pred = model.decode(z)

            generated_data.append(recon_norm.cpu().numpy()[0])
            predicted_max_vals.append(max_vals_pred.cpu().numpy()[0])

    return generated_data, predicted_max_vals


def generate_phase_space_comparison_figure(model, test_data, test_max_vals):
    """
    Generate comprehensive phase space comparison figure.

    Shows real (green) vs generated (blue) trajectories overlaid in same phase plots.
    Multiple species pairs across different panels.
    """
    print("\n" + "="*80)
    print("GENERATING PHASE SPACE COMPARISON FIGURE")
    print("="*80)

    # Get real samples
    np.random.seed(42)
    n_real = 30
    real_indices = np.random.choice(len(test_data), n_real, replace=False)

    real_samples = []
    for idx in real_indices:
        data_norm = test_data[idx]
        max_vals = test_max_vals[idx]
        data_original = data_norm * max_vals.reshape(1, -1, 1)
        real_samples.append(data_original[0])

    # Generate samples
    generated_data, predicted_max_vals = generate_samples(model, n_samples=n_real)

    gen_samples = []
    for i in range(n_real):
        data_norm = generated_data[i]
        max_vals = predicted_max_vals[i]
        data_original = data_norm * max_vals.reshape(1, -1, 1)
        gen_samples.append(data_original[0])

    # Create comprehensive figure with multiple phase projections
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Define species pairs for phase portraits
    phase_pairs = [
        (0, 1, 'Species 1 vs 2'),
        (0, 2, 'Species 1 vs 3'),
        (1, 2, 'Species 2 vs 3'),
        (2, 3, 'Species 3 vs 4'),
        (3, 4, 'Species 4 vs 5'),
        (4, 5, 'Species 5 vs 6'),
        (0, 6, 'Species 1 vs 7'),
        (3, 6, 'Species 4 vs 7'),
        (1, 5, 'Species 2 vs 6'),
    ]

    for plot_idx, (sp_i, sp_j, title) in enumerate(phase_pairs):
        ax = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])

        # Plot real trajectories (green, lighter)
        for real_sample in real_samples:
            ax.plot(real_sample[sp_i, :], real_sample[sp_j, :],
                   color='green', linewidth=1.0, alpha=0.15)

        # Plot generated trajectories (blue, lighter)
        for gen_sample in gen_samples:
            ax.plot(gen_sample[sp_i, :], gen_sample[sp_j, :],
                   color='blue', linewidth=1.0, alpha=0.15)

        # Plot one example of each with full opacity
        ax.plot(real_samples[0][sp_i, :], real_samples[0][sp_j, :],
               color='darkgreen', linewidth=2.5, alpha=0.9, label='Real', zorder=10)
        ax.plot(gen_samples[0][sp_i, :], gen_samples[0][sp_j, :],
               color='darkblue', linewidth=2.5, alpha=0.9, label='Generated', zorder=10)

        # Mark start points
        ax.scatter(real_samples[0][sp_i, 0], real_samples[0][sp_j, 0],
                  color='darkgreen', s=80, marker='o', edgecolor='black',
                  linewidth=1.5, zorder=15)
        ax.scatter(gen_samples[0][sp_i, 0], gen_samples[0][sp_j, 0],
                  color='darkblue', s=80, marker='o', edgecolor='black',
                  linewidth=1.5, zorder=15)

        ax.set_xlabel(f'Species {sp_i+1} Population', fontweight='bold', fontsize=10)
        ax.set_ylabel(f'Species {sp_j+1} Population', fontweight='bold', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(alpha=0.3)

        if plot_idx == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.95)

    plt.suptitle('Phase Space Comparison: Real vs Generated Trajectories (30 samples each)',
                fontweight='bold', fontsize=16, y=0.995)

    output_path = f'{OUTPUT_DIR}fig_phase_space_comparison_overlay.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_3d_phase_space_figure(model, test_data, test_max_vals):
    """
    Generate 3D phase space visualizations.
    """
    print("\n" + "="*80)
    print("GENERATING 3D PHASE SPACE FIGURE")
    print("="*80)

    from mpl_toolkits.mplot3d import Axes3D

    # Get samples
    np.random.seed(42)
    n_samples = 20

    # Real samples
    real_indices = np.random.choice(len(test_data), n_samples, replace=False)
    real_samples = []
    for idx in real_indices:
        data_norm = test_data[idx]
        max_vals = test_max_vals[idx]
        data_original = data_norm * max_vals.reshape(1, -1, 1)
        real_samples.append(data_original[0])

    # Generated samples
    generated_data, predicted_max_vals = generate_samples(model, n_samples=n_samples)
    gen_samples = []
    for i in range(n_samples):
        data_norm = generated_data[i]
        max_vals = predicted_max_vals[i]
        data_original = data_norm * max_vals.reshape(1, -1, 1)
        gen_samples.append(data_original[0])

    # Create figure with 3D subplots
    fig = plt.figure(figsize=(16, 10))

    # Define 3D projections
    projections_3d = [
        (0, 1, 2, 'Species 1-2-3'),
        (3, 4, 5, 'Species 4-5-6'),
        (0, 3, 6, 'Species 1-4-7'),
        (1, 4, 6, 'Species 2-5-7'),
    ]

    for plot_idx, (sp_i, sp_j, sp_k, title) in enumerate(projections_3d):
        ax = fig.add_subplot(2, 2, plot_idx + 1, projection='3d')

        # Plot real trajectories
        for real_sample in real_samples:
            ax.plot(real_sample[sp_i, :], real_sample[sp_j, :], real_sample[sp_k, :],
                   color='green', linewidth=1.0, alpha=0.3)

        # Plot generated trajectories
        for gen_sample in gen_samples:
            ax.plot(gen_sample[sp_i, :], gen_sample[sp_j, :], gen_sample[sp_k, :],
                   color='blue', linewidth=1.0, alpha=0.3)

        # Highlight one of each
        ax.plot(real_samples[0][sp_i, :], real_samples[0][sp_j, :], real_samples[0][sp_k, :],
               color='darkgreen', linewidth=2.5, alpha=0.9, label='Real')
        ax.plot(gen_samples[0][sp_i, :], gen_samples[0][sp_j, :], gen_samples[0][sp_k, :],
               color='darkblue', linewidth=2.5, alpha=0.9, label='Generated')

        ax.set_xlabel(f'Species {sp_i+1}', fontweight='bold', fontsize=10)
        ax.set_ylabel(f'Species {sp_j+1}', fontweight='bold', fontsize=10)
        ax.set_zlabel(f'Species {sp_k+1}', fontweight='bold', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)

        if plot_idx == 0:
            ax.legend(loc='best', fontsize=9)

    plt.suptitle('3D Phase Space: Real vs Generated Dynamics',
                fontweight='bold', fontsize=16)
    plt.tight_layout()

    output_path = f'{OUTPUT_DIR}fig_phase_space_3d_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate phase space comparison figures."""
    print("="*80)
    print("PHASE SPACE COMPARISON FIGURE GENERATION")
    print("Real vs Generated Trajectories")
    print("="*80)

    # Load model and data
    model, test_data, test_max_vals = load_model_and_data()

    # Generate 2D phase space comparison
    generate_phase_space_comparison_figure(model, test_data, test_max_vals)

    # Generate 3D phase space comparison
    generate_3d_phase_space_figure(model, test_data, test_max_vals)

    print("\n" + "="*80)
    print("PHASE SPACE COMPARISON FIGURES GENERATED!")
    print(f"Output: {OUTPUT_DIR}fig_phase_space_comparison_overlay.png/pdf")
    print(f"Output: {OUTPUT_DIR}fig_phase_space_3d_comparison.png/pdf")
    print("="*80)


if __name__ == "__main__":
    main()
