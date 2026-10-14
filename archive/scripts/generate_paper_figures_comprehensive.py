"""
Generate comprehensive paper-quality figures for Chaos, Solitons and Fractals journal.

This script creates multiple publication-ready visualizations:
1. Controlled generation examples showing diverse Lotka-Volterra dynamics
2. Scale control demonstration (same shape, different scales)
3. Latent space structure visualization (t-SNE/UMAP)
4. Latent space interpolation showing smooth transitions
5. Phase space trajectories (2D/3D projections)
6. Comparison: Real vs Generated dynamics
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
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

SPECIES_NAMES = ['Species 1', 'Species 2', 'Species 3', 'Species 4',
                 'Species 5', 'Species 6', 'Species 7']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2']


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


def generate_from_prior(model, n_samples=100):
    """
    Generate samples from the learned prior distribution.

    Returns:
        generated_data: List of generated time series (normalized)
        predicted_max_vals: List of predicted maximum values
    """
    print(f"\nGenerating {n_samples} samples from prior...")

    generated_data = []
    predicted_max_vals = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Generating"):
            # Sample from standard normal prior
            z = torch.randn(1, model_config['latent_dim']).to(DEVICE)

            # Decode
            recon_norm, max_vals_pred = model.decode(z)

            generated_data.append(recon_norm.cpu().numpy()[0])
            predicted_max_vals.append(max_vals_pred.cpu().numpy()[0])

    return generated_data, predicted_max_vals


def encode_to_latent_space(model, test_data, test_max_vals, n_samples=None):
    """Encode test samples to latent space."""
    if n_samples is None:
        n_samples = len(test_data)
    else:
        n_samples = min(n_samples, len(test_data))

    print(f"\nEncoding {n_samples} test samples to latent space...")

    latent_codes = []
    max_vals_list = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Encoding"):
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Encode
            x_rnn = x_norm.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(x_rnn)
            encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

            z_shape = model.shape_projector(encoded_summary)
            z_scale = model.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)

            mu = model.fc_mu(bottleneck_input)

            latent_codes.append(mu.cpu().numpy()[0])
            max_vals_list.append(test_max_vals[i])

    return np.array(latent_codes), np.array(max_vals_list)


# ============================================================================
# FIGURE 1: Controlled Generation Examples
# ============================================================================
def generate_figure_controlled_generation(model):
    """
    Show diverse, high-quality generated Lotka-Volterra dynamics.

    Display 6 generated examples with different characteristics.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 1: Controlled Generation Examples")
    print("="*80)

    # Generate samples
    generated_data, predicted_max_vals = generate_from_prior(model, n_samples=100)

    # Select diverse examples (different scales)
    max_val_sums = [np.sum(m) for m in predicted_max_vals]
    indices = np.linspace(0, len(max_val_sums)-1, 6, dtype=int)

    # Sort by total scale
    sorted_indices = np.argsort(max_val_sums)
    selected_indices = [sorted_indices[int(i)] for i in np.linspace(0, len(sorted_indices)-1, 6)]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    for idx, sample_idx in enumerate(selected_indices):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        data_norm = generated_data[sample_idx]
        max_vals = predicted_max_vals[sample_idx]

        # Reconstruct to original scale
        data_original = data_norm * max_vals.reshape(1, -1, 1)

        # Plot time series
        time = np.arange(65)
        for species_idx in range(7):
            ax.plot(time, data_original[0, species_idx, :],
                   label=SPECIES_NAMES[species_idx],
                   color=COLORS[species_idx], linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('Population', fontweight='bold')
        ax.set_title(f'Generated Sample {idx+1} (Scale: {np.mean(max_vals):.1f})',
                    fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    plt.suptitle('Controlled Generation: Diverse Lotka-Volterra Dynamics from Learned Prior',
                fontweight='bold', fontsize=15, y=0.995)

    output_path = f'{OUTPUT_DIR}fig_controlled_generation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 2: Scale Control Demonstration
# ============================================================================
def generate_figure_scale_control(model):
    """
    Demonstrate scale control: same shape at different scales.

    Take one latent code and decode with different predicted scales.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 2: Scale Control Demonstration")
    print("="*80)

    # Generate one sample
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, model_config['latent_dim']).to(DEVICE)
        recon_norm, max_vals_pred = model.decode(z)

        recon_norm_np = recon_norm.cpu().numpy()[0]
        max_vals_np = max_vals_pred.cpu().numpy()[0]

    # Create different scales (0.5x, 1x, 2x, 3x)
    scale_factors = [0.5, 1.0, 2.0, 3.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    time = np.arange(65)

    for idx, scale_factor in enumerate(scale_factors):
        ax = axes[idx]

        # Scale the maximum values
        scaled_max_vals = max_vals_np * scale_factor

        # Reconstruct with scaled max values
        data_scaled = recon_norm_np * scaled_max_vals.reshape(1, -1, 1)

        # Plot
        for species_idx in range(7):
            ax.plot(time, data_scaled[0, species_idx, :],
                   label=SPECIES_NAMES[species_idx],
                   color=COLORS[species_idx], linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('Population', fontweight='bold')
        ax.set_title(f'Scale Factor: {scale_factor}x (Mean Max: {np.mean(scaled_max_vals):.1f})',
                    fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    plt.suptitle('Scale Control: Same Dynamics Shape at Different Scales',
                fontweight='bold', fontsize=15)

    output_path = f'{OUTPUT_DIR}fig_scale_control.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 3: Latent Space Structure
# ============================================================================
def generate_figure_latent_space_structure(model, test_data, test_max_vals):
    """
    Visualize latent space structure using t-SNE and UMAP.

    Color by total scale (sum of max values).
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 3: Latent Space Structure")
    print("="*80)

    # Encode samples
    latent_codes, max_vals_array = encode_to_latent_space(
        model, test_data, test_max_vals, n_samples=3000
    )

    # Compute total scale for coloring
    total_scales = np.sum(max_vals_array, axis=1)

    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_codes)

    print("Computing UMAP projection...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    latent_umap = umap_reducer.fit_transform(latent_codes)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # t-SNE
    scatter1 = axes[0].scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                              c=total_scales, cmap='viridis', s=10, alpha=0.6)
    axes[0].set_xlabel('t-SNE Dimension 1', fontweight='bold')
    axes[0].set_ylabel('t-SNE Dimension 2', fontweight='bold')
    axes[0].set_title('t-SNE Projection of Latent Space', fontweight='bold', fontsize=13)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Total Scale (Sum of Max Values)', fontweight='bold')
    axes[0].grid(alpha=0.3)

    # UMAP
    scatter2 = axes[1].scatter(latent_umap[:, 0], latent_umap[:, 1],
                              c=total_scales, cmap='viridis', s=10, alpha=0.6)
    axes[1].set_xlabel('UMAP Dimension 1', fontweight='bold')
    axes[1].set_ylabel('UMAP Dimension 2', fontweight='bold')
    axes[1].set_title('UMAP Projection of Latent Space', fontweight='bold', fontsize=13)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Total Scale (Sum of Max Values)', fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Latent Space Structure: Scale-Conditioned CVAE',
                fontweight='bold', fontsize=15)

    output_path = f'{OUTPUT_DIR}fig_latent_space_structure.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 4: Latent Space Interpolation
# ============================================================================
def generate_figure_latent_interpolation(model, test_data, test_max_vals):
    """
    Show smooth interpolation between two different dynamics.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 4: Latent Space Interpolation")
    print("="*80)

    # Encode two diverse samples
    latent_codes, max_vals_array = encode_to_latent_space(
        model, test_data, test_max_vals, n_samples=len(test_data)
    )

    # Select two samples with very different scales
    total_scales = np.sum(max_vals_array, axis=1)
    idx_small = np.argmin(total_scales)
    idx_large = np.argmax(total_scales)

    z1 = latent_codes[idx_small]
    z2 = latent_codes[idx_large]

    # Interpolate
    n_steps = 6
    alphas = np.linspace(0, 1, n_steps)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    time = np.arange(65)

    model.eval()
    with torch.no_grad():
        for idx, alpha in enumerate(alphas):
            # Interpolate in latent space
            z_interp = (1 - alpha) * z1 + alpha * z2
            z_interp_torch = torch.tensor(z_interp, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Decode
            recon_norm, max_vals_pred = model.decode(z_interp_torch)

            recon_norm_np = recon_norm.cpu().numpy()[0]
            max_vals_np = max_vals_pred.cpu().numpy()[0]

            # Reconstruct to original scale
            data_original = recon_norm_np * max_vals_np.reshape(1, -1, 1)

            # Plot
            ax = axes[idx]
            for species_idx in range(7):
                ax.plot(time, data_original[0, species_idx, :],
                       label=SPECIES_NAMES[species_idx],
                       color=COLORS[species_idx], linewidth=1.5, alpha=0.8)

            ax.set_xlabel('Time Step', fontweight='bold')
            ax.set_ylabel('Population', fontweight='bold')
            ax.set_title(f'α = {alpha:.2f} (Scale: {np.mean(max_vals_np):.1f})',
                        fontweight='bold')
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)

    plt.suptitle('Latent Space Interpolation: Smooth Transitions Between Dynamics',
                fontweight='bold', fontsize=15)

    output_path = f'{OUTPUT_DIR}fig_latent_interpolation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 5: Phase Space Trajectories
# ============================================================================
def generate_figure_phase_space(model):
    """
    Show 2D phase space projections of generated dynamics.

    Display trajectories in different 2D projections.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 5: Phase Space Trajectories")
    print("="*80)

    # Generate samples
    generated_data, predicted_max_vals = generate_from_prior(model, n_samples=50)

    # Select 4 diverse samples
    max_val_sums = [np.sum(m) for m in predicted_max_vals]
    sorted_indices = np.argsort(max_val_sums)
    selected_indices = [sorted_indices[int(i)] for i in np.linspace(0, len(sorted_indices)-1, 4)]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Phase space pairs to plot
    phase_pairs = [(0, 1), (2, 3), (4, 5), (0, 6)]
    pair_labels = [('Species 1', 'Species 2'),
                   ('Species 3', 'Species 4'),
                   ('Species 5', 'Species 6'),
                   ('Species 1', 'Species 7')]

    for idx, (sample_idx, (i, j), (label_i, label_j)) in enumerate(zip(selected_indices, phase_pairs, pair_labels)):
        ax = axes[idx]

        data_norm = generated_data[sample_idx]
        max_vals = predicted_max_vals[sample_idx]

        # Reconstruct to original scale
        data_original = data_norm * max_vals.reshape(1, -1, 1)

        # Extract trajectories
        traj_i = data_original[0, i, :]
        traj_j = data_original[0, j, :]

        # Plot phase space trajectory
        ax.plot(traj_i, traj_j, color='steelblue', linewidth=2, alpha=0.7)
        ax.scatter(traj_i[0], traj_j[0], color='green', s=100, marker='o',
                  label='Start', zorder=5, edgecolor='black', linewidth=1.5)
        ax.scatter(traj_i[-1], traj_j[-1], color='red', s=100, marker='s',
                  label='End', zorder=5, edgecolor='black', linewidth=1.5)

        ax.set_xlabel(label_i, fontweight='bold')
        ax.set_ylabel(label_j, fontweight='bold')
        ax.set_title(f'Phase Portrait: {label_i} vs {label_j}', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

    plt.suptitle('Phase Space Trajectories of Generated Lotka-Volterra Dynamics',
                fontweight='bold', fontsize=15)
    plt.tight_layout()

    output_path = f'{OUTPUT_DIR}fig_phase_space.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 6: Real vs Generated Comparison
# ============================================================================
def generate_figure_real_vs_generated(model, test_data, test_max_vals):
    """
    Side-by-side comparison of real and generated dynamics.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 6: Real vs Generated Comparison")
    print("="*80)

    # Select 3 random real samples
    np.random.seed(42)
    real_indices = np.random.choice(len(test_data), 3, replace=False)

    # Generate 3 samples from prior
    generated_data, predicted_max_vals = generate_from_prior(model, n_samples=3)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    time = np.arange(65)

    for row in range(3):
        # Real sample (left column)
        ax_real = axes[row, 0]
        real_idx = real_indices[row]
        data_norm_real = test_data[real_idx]
        max_vals_real = test_max_vals[real_idx]
        data_original_real = data_norm_real * max_vals_real.reshape(1, -1, 1)

        for species_idx in range(7):
            ax_real.plot(time, data_original_real[0, species_idx, :],
                        label=SPECIES_NAMES[species_idx],
                        color=COLORS[species_idx], linewidth=1.5, alpha=0.8)

        ax_real.set_xlabel('Time Step', fontweight='bold')
        ax_real.set_ylabel('Population', fontweight='bold')
        ax_real.set_title(f'Real Sample {row+1} (Scale: {np.mean(max_vals_real):.1f})',
                         fontweight='bold', color='darkgreen')
        ax_real.grid(alpha=0.3)
        if row == 0:
            ax_real.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

        # Generated sample (right column)
        ax_gen = axes[row, 1]
        data_norm_gen = generated_data[row]
        max_vals_gen = predicted_max_vals[row]
        data_original_gen = data_norm_gen * max_vals_gen.reshape(1, -1, 1)

        for species_idx in range(7):
            ax_gen.plot(time, data_original_gen[0, species_idx, :],
                       label=SPECIES_NAMES[species_idx],
                       color=COLORS[species_idx], linewidth=1.5, alpha=0.8)

        ax_gen.set_xlabel('Time Step', fontweight='bold')
        ax_gen.set_ylabel('Population', fontweight='bold')
        ax_gen.set_title(f'Generated Sample {row+1} (Scale: {np.mean(max_vals_gen):.1f})',
                        fontweight='bold', color='darkblue')
        ax_gen.grid(alpha=0.3)
        if row == 0:
            ax_gen.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Column titles
    axes[0, 0].text(0.5, 1.15, 'REAL DATA', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', ha='center', color='darkgreen')
    axes[0, 1].text(0.5, 1.15, 'GENERATED DATA', transform=axes[0, 1].transAxes,
                   fontsize=14, fontweight='bold', ha='center', color='darkblue')

    plt.suptitle('Qualitative Comparison: Real vs Generated Lotka-Volterra Dynamics',
                fontweight='bold', fontsize=15, y=0.995)
    plt.tight_layout()

    output_path = f'{OUTPUT_DIR}fig_real_vs_generated.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Generate all paper figures."""
    print("="*80)
    print("COMPREHENSIVE PAPER FIGURE GENERATION")
    print("For: Chaos, Solitons and Fractals")
    print("Scale-Conditioned CVAE for Lotka-Volterra Dynamics")
    print("="*80)

    # Load model and data
    model, test_data, test_max_vals = load_model_and_data()

    # Generate all figures
    generate_figure_controlled_generation(model)
    generate_figure_scale_control(model)
    generate_figure_latent_space_structure(model, test_data, test_max_vals)
    generate_figure_latent_interpolation(model, test_data, test_max_vals)
    generate_figure_phase_space(model)
    generate_figure_real_vs_generated(model, test_data, test_max_vals)

    print("\n" + "="*80)
    print("ALL PAPER FIGURES GENERATED SUCCESSFULLY!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    print("\nGenerated figures:")
    print("  1. fig_controlled_generation.png/pdf - Diverse generated dynamics")
    print("  2. fig_scale_control.png/pdf - Same shape at different scales")
    print("  3. fig_latent_space_structure.png/pdf - t-SNE & UMAP projections")
    print("  4. fig_latent_interpolation.png/pdf - Smooth latent transitions")
    print("  5. fig_phase_space.png/pdf - Phase space trajectories")
    print("  6. fig_real_vs_generated.png/pdf - Real vs generated comparison")
    print("="*80)


if __name__ == "__main__":
    main()
