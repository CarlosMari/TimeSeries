"""
Generate recurrence plot and dynamical complexity analysis figure.

Highly relevant for Chaos, Solitons and Fractals journal.

This script creates:
1. Recurrence plots comparing real vs generated dynamics
2. Correlation sum analysis
3. Poincaré sections in phase space
4. Power spectrum analysis
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, squareform
from scipy import signal
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


def compute_recurrence_matrix(time_series, threshold=None, eps_percentile=10):
    """
    Compute recurrence matrix for a multivariate time series.

    Args:
        time_series: shape (n_species, seq_len)
        threshold: Distance threshold (if None, use percentile)
        eps_percentile: Percentile for automatic threshold

    Returns:
        Recurrence matrix (seq_len, seq_len)
    """
    # Transpose to (seq_len, n_species)
    trajectory = time_series.T

    # Compute pairwise distances
    distances = squareform(pdist(trajectory, metric='euclidean'))

    # Set threshold
    if threshold is None:
        threshold = np.percentile(distances, eps_percentile)

    # Create recurrence matrix
    recurrence = (distances < threshold).astype(float)

    return recurrence, distances


def compute_power_spectrum(time_series):
    """
    Compute power spectrum for each species.

    Args:
        time_series: shape (n_species, seq_len)

    Returns:
        frequencies, power_spectra (n_species, n_freq)
    """
    n_species, seq_len = time_series.shape
    power_spectra = []

    for i in range(n_species):
        # Compute FFT
        fft = np.fft.fft(time_series[i, :])
        freqs = np.fft.fftfreq(seq_len)

        # Power spectrum (one-sided)
        power = np.abs(fft) ** 2
        power_spectra.append(power[:seq_len // 2])

    freqs = freqs[:seq_len // 2]

    return freqs, np.array(power_spectra)


def generate_comparison_figure(model, test_data, test_max_vals):
    """
    Generate comprehensive dynamical complexity comparison figure.

    Compare real vs generated samples using:
    - Recurrence plots
    - Power spectra
    - Phase space projections
    - Correlation analysis
    """
    print("\n" + "="*80)
    print("GENERATING RECURRENCE & DYNAMICAL COMPLEXITY FIGURE")
    print("="*80)

    # Select one real sample
    np.random.seed(42)
    real_idx = np.random.randint(0, len(test_data))

    data_norm_real = test_data[real_idx]
    max_vals_real = test_max_vals[real_idx]
    data_real = data_norm_real * max_vals_real.reshape(1, -1, 1)

    # Generate one sample
    print("Generating sample from prior...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, model_config['latent_dim']).to(DEVICE)
        recon_norm, max_vals_pred = model.decode(z)

        data_norm_gen = recon_norm.cpu().numpy()[0]
        max_vals_gen = max_vals_pred.cpu().numpy()[0]
        data_gen = data_norm_gen * max_vals_gen.reshape(1, -1, 1)

    print("Computing recurrence matrices...")
    recurrence_real, distances_real = compute_recurrence_matrix(data_real[0])
    recurrence_gen, distances_gen = compute_recurrence_matrix(data_gen[0])

    print("Computing power spectra...")
    freqs_real, power_real = compute_power_spectrum(data_real[0])
    freqs_gen, power_gen = compute_power_spectrum(data_gen[0])

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.35)

    # ========== ROW 1: Time Series ==========
    ax_time_real = fig.add_subplot(gs[0, 0:2])
    ax_time_gen = fig.add_subplot(gs[0, 2])

    time = np.arange(65)
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    # Real time series
    for i in range(7):
        ax_time_real.plot(time, data_real[0, i, :], color=colors[i],
                         linewidth=1.5, alpha=0.8, label=f'Species {i+1}')
    ax_time_real.set_xlabel('Time Step', fontweight='bold')
    ax_time_real.set_ylabel('Population', fontweight='bold')
    ax_time_real.set_title('Real Lotka-Volterra Dynamics', fontweight='bold', fontsize=12, color='darkgreen')
    ax_time_real.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    ax_time_real.grid(alpha=0.3)

    # Generated time series
    for i in range(7):
        ax_time_gen.plot(time, data_gen[0, i, :], color=colors[i],
                        linewidth=1.5, alpha=0.8, label=f'Species {i+1}')
    ax_time_gen.set_xlabel('Time Step', fontweight='bold')
    ax_time_gen.set_ylabel('Population', fontweight='bold')
    ax_time_gen.set_title('Generated Dynamics', fontweight='bold', fontsize=12, color='darkblue')
    ax_time_gen.grid(alpha=0.3)

    # ========== ROW 2: Recurrence Plots ==========
    ax_rec_real = fig.add_subplot(gs[1, 0])
    ax_rec_gen = fig.add_subplot(gs[1, 1])
    ax_phase = fig.add_subplot(gs[1, 2])

    # Real recurrence plot
    im1 = ax_rec_real.imshow(recurrence_real, cmap='binary', origin='lower',
                             interpolation='nearest', aspect='auto')
    ax_rec_real.set_xlabel('Time Step', fontweight='bold')
    ax_rec_real.set_ylabel('Time Step', fontweight='bold')
    ax_rec_real.set_title('Recurrence Plot (Real)', fontweight='bold', fontsize=12, color='darkgreen')
    plt.colorbar(im1, ax=ax_rec_real, fraction=0.046, pad=0.04)

    # Generated recurrence plot
    im2 = ax_rec_gen.imshow(recurrence_gen, cmap='binary', origin='lower',
                            interpolation='nearest', aspect='auto')
    ax_rec_gen.set_xlabel('Time Step', fontweight='bold')
    ax_rec_gen.set_ylabel('Time Step', fontweight='bold')
    ax_rec_gen.set_title('Recurrence Plot (Generated)', fontweight='bold', fontsize=12, color='darkblue')
    plt.colorbar(im2, ax=ax_rec_gen, fraction=0.046, pad=0.04)

    # Phase space projection (Species 1 vs Species 2)
    ax_phase.plot(data_real[0, 0, :], data_real[0, 1, :], color='darkgreen',
                 linewidth=2, alpha=0.7, label='Real')
    ax_phase.plot(data_gen[0, 0, :], data_gen[0, 1, :], color='darkblue',
                 linewidth=2, alpha=0.7, label='Generated')
    ax_phase.scatter(data_real[0, 0, 0], data_real[0, 1, 0], color='green',
                    s=100, marker='o', edgecolor='black', linewidth=1.5, zorder=5)
    ax_phase.scatter(data_gen[0, 0, 0], data_gen[0, 1, 0], color='blue',
                    s=100, marker='o', edgecolor='black', linewidth=1.5, zorder=5)
    ax_phase.set_xlabel('Species 1 Population', fontweight='bold')
    ax_phase.set_ylabel('Species 2 Population', fontweight='bold')
    ax_phase.set_title('Phase Space Projection', fontweight='bold', fontsize=12)
    ax_phase.legend(loc='best', fontsize=9, framealpha=0.9)
    ax_phase.grid(alpha=0.3)

    # ========== ROW 3: Power Spectra and Statistics ==========
    ax_power = fig.add_subplot(gs[2, 0:2])
    ax_stats = fig.add_subplot(gs[2, 2])

    # Power spectra (average across species)
    power_real_avg = np.mean(power_real, axis=0)
    power_gen_avg = np.mean(power_gen, axis=0)

    ax_power.semilogy(freqs_real, power_real_avg, color='darkgreen',
                     linewidth=2, alpha=0.7, label='Real')
    ax_power.semilogy(freqs_gen, power_gen_avg, color='darkblue',
                     linewidth=2, alpha=0.7, label='Generated')
    ax_power.set_xlabel('Frequency', fontweight='bold')
    ax_power.set_ylabel('Power (log scale)', fontweight='bold')
    ax_power.set_title('Average Power Spectrum', fontweight='bold', fontsize=12)
    ax_power.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_power.grid(alpha=0.3)

    # Statistics comparison
    ax_stats.axis('off')

    # Compute statistics
    stats_text = "Dynamical Complexity Metrics\n" + "="*35 + "\n\n"

    # Recurrence rate
    rr_real = np.sum(recurrence_real) / (recurrence_real.shape[0] ** 2)
    rr_gen = np.sum(recurrence_gen) / (recurrence_gen.shape[0] ** 2)
    stats_text += f"Recurrence Rate:\n"
    stats_text += f"  Real:      {rr_real:.4f}\n"
    stats_text += f"  Generated: {rr_gen:.4f}\n\n"

    # Mean distance
    mean_dist_real = np.mean(distances_real[np.triu_indices_from(distances_real, k=1)])
    mean_dist_gen = np.mean(distances_gen[np.triu_indices_from(distances_gen, k=1)])
    stats_text += f"Mean Trajectory Distance:\n"
    stats_text += f"  Real:      {mean_dist_real:.4f}\n"
    stats_text += f"  Generated: {mean_dist_gen:.4f}\n\n"

    # Variance
    var_real = np.var(data_real)
    var_gen = np.var(data_gen)
    stats_text += f"Total Variance:\n"
    stats_text += f"  Real:      {var_real:.4f}\n"
    stats_text += f"  Generated: {var_gen:.4f}\n\n"

    # Dominant frequency
    dom_freq_real = freqs_real[np.argmax(power_real_avg[1:])+1]  # Skip DC component
    dom_freq_gen = freqs_gen[np.argmax(power_gen_avg[1:])+1]
    stats_text += f"Dominant Frequency:\n"
    stats_text += f"  Real:      {dom_freq_real:.4f}\n"
    stats_text += f"  Generated: {dom_freq_gen:.4f}\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Main title
    plt.suptitle('Dynamical Complexity Analysis: Real vs Generated Lotka-Volterra Systems',
                fontweight='bold', fontsize=16, y=0.995)

    output_path = f'{OUTPUT_DIR}fig_recurrence_dynamics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    """Generate recurrence and dynamical complexity figure."""
    print("="*80)
    print("RECURRENCE & DYNAMICAL COMPLEXITY FIGURE GENERATION")
    print("For: Chaos, Solitons and Fractals")
    print("="*80)

    # Load model and data
    model, test_data, test_max_vals = load_model_and_data()

    # Generate comparison figure
    generate_comparison_figure(model, test_data, test_max_vals)

    print("\n" + "="*80)
    print("RECURRENCE FIGURE GENERATED SUCCESSFULLY!")
    print(f"Output: {OUTPUT_DIR}fig_recurrence_dynamics.png/pdf")
    print("="*80)


if __name__ == "__main__":
    main()
