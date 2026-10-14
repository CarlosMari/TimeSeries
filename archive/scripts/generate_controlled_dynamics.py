"""
Controlled generation: Map latent space to dynamical properties.

This script:
1. Samples many latent codes and generates dynamics
2. Analyzes dynamical properties (oscillations, stability, variance, etc.)
3. Maps latent space regions to specific behaviors
4. Demonstrates controlled generation of desired dynamics
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
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
MODEL_PATH = 'model_ckpts/model_final_50_conditioned.pth'
OUTPUT_DIR = 'final figures/'

SPECIES_NAMES = ['Species 1', 'Species 2', 'Species 3', 'Species 4',
                 'Species 5', 'Species 6', 'Species 7']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2']


def load_model():
    """Load trained model."""
    print("Loading model...")
    model = LSTM_VAE(config=model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded")
    return model


def analyze_oscillation_strength(time_series):
    """
    Measure oscillation strength using FFT.

    Returns:
        oscillation_score: Higher = more oscillatory behavior
        dominant_freq: Dominant frequency component
    """
    # Average across species
    avg_signal = np.mean(time_series, axis=0)

    # FFT
    n = len(avg_signal)
    yf = fft(avg_signal)
    xf = fftfreq(n)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Skip DC component, find dominant frequency
    if len(power) > 1:
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = xf[dominant_idx]
        oscillation_score = power[dominant_idx]
    else:
        dominant_freq = 0
        oscillation_score = 0

    return oscillation_score, dominant_freq


def analyze_stability(time_series):
    """
    Measure stability/chaos using variance in derivatives.

    Returns:
        stability_score: Higher = more chaotic/unstable
    """
    # Compute derivatives
    derivatives = np.diff(time_series, axis=1)

    # Variance of derivatives (averaged across species)
    stability_score = np.mean(np.var(derivatives, axis=1))

    return stability_score


def analyze_complexity(time_series):
    """
    Measure dynamical complexity.

    Returns:
        complexity_score: Based on variance and oscillations
    """
    # Total variance
    total_var = np.var(time_series)

    # Number of direction changes (averaged across species)
    direction_changes = 0
    for i in range(time_series.shape[0]):
        diff = np.diff(time_series[i, :])
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        direction_changes += sign_changes

    direction_changes /= time_series.shape[0]

    complexity_score = total_var * (1 + direction_changes / 10)

    return complexity_score


def generate_and_analyze(model, n_samples=500):
    """
    Generate samples and analyze their dynamical properties.

    Returns:
        latent_codes: (n_samples, latent_dim)
        properties: dict of arrays with dynamical metrics
        generated_data: list of generated time series
    """
    print(f"\nGenerating and analyzing {n_samples} samples...")

    latent_codes = []
    oscillation_scores = []
    dominant_freqs = []
    stability_scores = []
    complexity_scores = []
    total_variances = []
    mean_populations = []
    generated_data = []
    predicted_max_vals = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Generating & analyzing"):
            # Sample latent code
            z = torch.randn(1, model_config['latent_dim']).to(DEVICE)

            # Generate
            recon_norm, max_vals_pred = model.decode(z)

            # Convert to original scale
            recon_norm_np = recon_norm.cpu().numpy()[0]
            max_vals_np = max_vals_pred.cpu().numpy()[0]
            data_original = recon_norm_np * max_vals_np.reshape(1, -1, 1)

            # Analyze
            osc_score, dom_freq = analyze_oscillation_strength(data_original[0])
            stab_score = analyze_stability(data_original[0])
            comp_score = analyze_complexity(data_original[0])
            total_var = np.var(data_original)
            mean_pop = np.mean(data_original)

            # Store
            latent_codes.append(z.cpu().numpy()[0])
            oscillation_scores.append(osc_score)
            dominant_freqs.append(dom_freq)
            stability_scores.append(stab_score)
            complexity_scores.append(comp_score)
            total_variances.append(total_var)
            mean_populations.append(mean_pop)
            generated_data.append(data_original[0])
            predicted_max_vals.append(max_vals_np)

    properties = {
        'oscillation': np.array(oscillation_scores),
        'frequency': np.array(dominant_freqs),
        'stability': np.array(stability_scores),
        'complexity': np.array(complexity_scores),
        'variance': np.array(total_variances),
        'mean_population': np.array(mean_populations),
    }

    return np.array(latent_codes), properties, generated_data, predicted_max_vals


def generate_controlled_generation_figure(latent_codes, properties, generated_data, predicted_max_vals):
    """
    Create comprehensive controlled generation visualization.
    """
    print("\n" + "="*80)
    print("GENERATING CONTROLLED GENERATION FIGURE")
    print("="*80)

    # PCA on latent codes for visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_codes)

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ========== Panel 1: Latent space colored by oscillation ==========
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=properties['oscillation'], cmap='viridis',
                          s=20, alpha=0.6)
    ax1.set_xlabel('Latent PC 1', fontweight='bold')
    ax1.set_ylabel('Latent PC 2', fontweight='bold')
    ax1.set_title('Oscillation Strength', fontweight='bold', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Oscillation Score', fontweight='bold')
    ax1.grid(alpha=0.3)

    # ========== Panel 2: Latent space colored by complexity ==========
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=properties['complexity'], cmap='plasma',
                          s=20, alpha=0.6)
    ax2.set_xlabel('Latent PC 1', fontweight='bold')
    ax2.set_ylabel('Latent PC 2', fontweight='bold')
    ax2.set_title('Dynamical Complexity', fontweight='bold', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Complexity Score', fontweight='bold')
    ax2.grid(alpha=0.3)

    # ========== Panel 3: Latent space colored by stability ==========
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=properties['stability'], cmap='coolwarm',
                          s=20, alpha=0.6)
    ax3.set_xlabel('Latent PC 1', fontweight='bold')
    ax3.set_ylabel('Latent PC 2', fontweight='bold')
    ax3.set_title('Stability/Chaos', fontweight='bold', fontsize=12)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Stability Score', fontweight='bold')
    ax3.grid(alpha=0.3)

    # ========== Panels 4-6: Controlled examples ==========
    # Find examples with specific properties

    # High oscillation
    high_osc_idx = np.argsort(properties['oscillation'])[-1]
    ax4 = fig.add_subplot(gs[1, 0])
    plot_dynamics(ax4, generated_data[high_osc_idx], predicted_max_vals[high_osc_idx],
                 f"High Oscillation (score={properties['oscillation'][high_osc_idx]:.2f})")

    # High complexity
    high_comp_idx = np.argsort(properties['complexity'])[-1]
    ax5 = fig.add_subplot(gs[1, 1])
    plot_dynamics(ax5, generated_data[high_comp_idx], predicted_max_vals[high_comp_idx],
                 f"High Complexity (score={properties['complexity'][high_comp_idx]:.2f})")

    # High stability (chaotic)
    high_stab_idx = np.argsort(properties['stability'])[-1]
    ax6 = fig.add_subplot(gs[1, 2])
    plot_dynamics(ax6, generated_data[high_stab_idx], predicted_max_vals[high_stab_idx],
                 f"High Chaos (score={properties['stability'][high_stab_idx]:.2f})")

    # ========== Panels 7-9: More controlled examples ==========

    # Low oscillation (smooth)
    low_osc_idx = np.argsort(properties['oscillation'])[0]
    ax7 = fig.add_subplot(gs[2, 0])
    plot_dynamics(ax7, generated_data[low_osc_idx], predicted_max_vals[low_osc_idx],
                 f"Low Oscillation (score={properties['oscillation'][low_osc_idx]:.2f})")

    # Medium complexity
    median_comp_idx = np.argsort(properties['complexity'])[len(properties['complexity'])//2]
    ax8 = fig.add_subplot(gs[2, 1])
    plot_dynamics(ax8, generated_data[median_comp_idx], predicted_max_vals[median_comp_idx],
                 f"Medium Complexity (score={properties['complexity'][median_comp_idx]:.2f})")

    # Low stability (stable)
    low_stab_idx = np.argsort(properties['stability'])[0]
    ax9 = fig.add_subplot(gs[2, 2])
    plot_dynamics(ax9, generated_data[low_stab_idx], predicted_max_vals[low_stab_idx],
                 f"Low Chaos (score={properties['stability'][low_stab_idx]:.2f})")

    plt.suptitle('Controlled Generation: Mapping Latent Space to Dynamical Properties',
                fontweight='bold', fontsize=16, y=0.995)

    output_path = f'{OUTPUT_DIR}fig_controlled_generation_properties.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_dynamics(ax, data, max_vals, title):
    """Helper to plot dynamics on axis."""
    time = np.arange(65)
    for species_idx in range(7):
        ax.plot(time, data[species_idx, :],
               color=COLORS[species_idx], linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time Step', fontweight='bold', fontsize=9)
    ax.set_ylabel('Population', fontweight='bold', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)


def generate_property_distributions_figure(properties):
    """
    Show distributions of dynamical properties.
    """
    print("\n" + "="*80)
    print("GENERATING PROPERTY DISTRIBUTIONS FIGURE")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    property_names = ['oscillation', 'complexity', 'stability',
                     'variance', 'frequency', 'mean_population']
    titles = ['Oscillation Strength', 'Dynamical Complexity', 'Stability/Chaos',
             'Total Variance', 'Dominant Frequency', 'Mean Population']

    for idx, (prop_name, title) in enumerate(zip(property_names, titles)):
        ax = axes[idx]

        values = properties[prop_name]

        # Histogram
        ax.hist(values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')

        # Statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.3f}')

        ax.set_xlabel(title, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'{title} Distribution', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Distribution of Dynamical Properties in Generated Samples (N=500)',
                fontweight='bold', fontsize=15)
    plt.tight_layout()

    output_path = f'{OUTPUT_DIR}fig_property_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_selective_sampling_guide(latent_codes, properties):
    """
    Create a guide showing how to selectively sample latent codes
    for desired dynamical properties.
    """
    print("\n" + "="*80)
    print("GENERATING SELECTIVE SAMPLING GUIDE")
    print("="*80)

    # Find regions in latent space with specific properties
    high_osc_indices = np.argsort(properties['oscillation'])[-50:]
    low_osc_indices = np.argsort(properties['oscillation'])[:50]
    high_comp_indices = np.argsort(properties['complexity'])[-50:]

    high_osc_latents = latent_codes[high_osc_indices]
    low_osc_latents = latent_codes[low_osc_indices]
    high_comp_latents = latent_codes[high_comp_indices]

    # Compute statistics
    high_osc_mean = np.mean(high_osc_latents, axis=0)
    high_osc_std = np.std(high_osc_latents, axis=0)

    low_osc_mean = np.mean(low_osc_latents, axis=0)
    low_osc_std = np.std(low_osc_latents, axis=0)

    high_comp_mean = np.mean(high_comp_latents, axis=0)
    high_comp_std = np.std(high_comp_latents, axis=0)

    # Save as sampling guide
    sampling_guide = {
        'high_oscillation': {
            'mean': high_osc_mean,
            'std': high_osc_std,
            'example_score': properties['oscillation'][high_osc_indices[-1]]
        },
        'low_oscillation': {
            'mean': low_osc_mean,
            'std': low_osc_std,
            'example_score': properties['oscillation'][low_osc_indices[0]]
        },
        'high_complexity': {
            'mean': high_comp_mean,
            'std': high_comp_std,
            'example_score': properties['complexity'][high_comp_indices[-1]]
        }
    }

    # Save to file
    with open(f'{OUTPUT_DIR}latent_sampling_guide.pkl', 'wb') as f:
        pickle.dump(sampling_guide, f)

    print(f"Saved sampling guide to: {OUTPUT_DIR}latent_sampling_guide.pkl")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot mean latent codes for different properties
    dims = np.arange(50)

    ax = axes[0]
    ax.plot(dims, high_osc_mean, color='red', linewidth=2, label='High Oscillation', marker='o', markersize=3)
    ax.fill_between(dims, high_osc_mean - high_osc_std, high_osc_mean + high_osc_std,
                    color='red', alpha=0.2)
    ax.set_xlabel('Latent Dimension', fontweight='bold')
    ax.set_ylabel('Mean Value', fontweight='bold')
    ax.set_title('High Oscillation Region', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    ax = axes[1]
    ax.plot(dims, low_osc_mean, color='blue', linewidth=2, label='Low Oscillation', marker='o', markersize=3)
    ax.fill_between(dims, low_osc_mean - low_osc_std, low_osc_mean + low_osc_std,
                    color='blue', alpha=0.2)
    ax.set_xlabel('Latent Dimension', fontweight='bold')
    ax.set_ylabel('Mean Value', fontweight='bold')
    ax.set_title('Low Oscillation Region', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    ax = axes[2]
    ax.plot(dims, high_comp_mean, color='purple', linewidth=2, label='High Complexity', marker='o', markersize=3)
    ax.fill_between(dims, high_comp_mean - high_comp_std, high_comp_mean + high_comp_std,
                    color='purple', alpha=0.2)
    ax.set_xlabel('Latent Dimension', fontweight='bold')
    ax.set_ylabel('Mean Value', fontweight='bold')
    ax.set_title('High Complexity Region', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    plt.suptitle('Latent Space Regions for Controlled Generation',
                fontweight='bold', fontsize=15)
    plt.tight_layout()

    output_path = f'{OUTPUT_DIR}fig_latent_sampling_regions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print instructions
    print("\n" + "="*80)
    print("HOW TO USE CONTROLLED GENERATION:")
    print("="*80)
    print("\nTo generate dynamics with HIGH OSCILLATION:")
    print("  z = torch.randn(1, 50) * torch.tensor(high_osc_std) + torch.tensor(high_osc_mean)")
    print(f"  Expected oscillation score: ~{sampling_guide['high_oscillation']['example_score']:.2f}")
    print("\nTo generate dynamics with LOW OSCILLATION:")
    print("  z = torch.randn(1, 50) * torch.tensor(low_osc_std) + torch.tensor(low_osc_mean)")
    print(f"  Expected oscillation score: ~{sampling_guide['low_oscillation']['example_score']:.2f}")
    print("\nTo generate dynamics with HIGH COMPLEXITY:")
    print("  z = torch.randn(1, 50) * torch.tensor(high_comp_std) + torch.tensor(high_comp_mean)")
    print(f"  Expected complexity score: ~{sampling_guide['high_complexity']['example_score']:.2f}")
    print("="*80)


def main():
    """Main execution."""
    print("="*80)
    print("CONTROLLED GENERATION ANALYSIS")
    print("Mapping Latent Space to Dynamical Properties")
    print("="*80)

    # Load model
    model = load_model()

    # Generate and analyze samples
    latent_codes, properties, generated_data, predicted_max_vals = generate_and_analyze(model, n_samples=500)

    # Generate visualizations
    generate_controlled_generation_figure(latent_codes, properties, generated_data, predicted_max_vals)
    generate_property_distributions_figure(properties)
    generate_selective_sampling_guide(latent_codes, properties)

    print("\n" + "="*80)
    print("CONTROLLED GENERATION ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  • {OUTPUT_DIR}fig_controlled_generation_properties.png/pdf")
    print(f"  • {OUTPUT_DIR}fig_property_distributions.png/pdf")
    print(f"  • {OUTPUT_DIR}fig_latent_sampling_regions.png/pdf")
    print(f"  • {OUTPUT_DIR}latent_sampling_guide.pkl")
    print("="*80)


if __name__ == "__main__":
    main()
