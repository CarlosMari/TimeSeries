"""
Latent Space Interpolation: Stable → Oscillatory Dynamics

Professional figure demonstrating smooth interpolation between
contrasting ecological dynamics for Chaos, Solitons and Fractals.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Professional journal style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE
from utils.post_processing import apply_extinction_threshold

# Configuration
MODEL_CONFIG = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'
OUTPUT_DIR = 'final figures/'

N_INTERPOLATION_STEPS = 7
COLORS = sns.color_palette("husl", 7)


def compute_oscillation_strength(trajectory):
    """Compute oscillation strength as mean temporal std across species."""
    return np.mean([np.std(trajectory[sp, :]) for sp in range(trajectory.shape[0])])


def find_extreme_samples(test_data, test_max_vals, model, n_candidates=2000):
    """Find stable sample from test set and load Rank #5 from saved candidates."""

    # Find stable sample from test set
    print("\nFinding stable sample from test set...")
    oscillation_scores = []

    with torch.no_grad():
        for i in tqdm(range(min(1000, len(test_data))), desc="Finding stable"):
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Get reconstruction
            recon, _, _, max_pred, _ = model(x_norm, max_vals, teacher_forcing_ratio=0.0)
            recon_denorm = recon.cpu().numpy()[0] * max_pred.cpu().numpy()[0][:, np.newaxis]
            recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T

            # Compute oscillation
            osc = compute_oscillation_strength(recon_fixed)
            oscillation_scores.append((i, osc, recon_fixed))

    # Sort by oscillation to get most stable
    oscillation_scores.sort(key=lambda x: x[1])
    stable_idx, stable_osc, stable_traj = oscillation_scores[0]
    print(f"  Stable sample: idx={stable_idx}, oscillation={stable_osc:.4f}")

    # Load saved Rank #5 ultra-oscillatory sample
    print("\nLoading saved Rank #5 ultra-oscillatory sample...")
    with open('data/ultra_oscillatory_candidates.pkl', 'rb') as f:
        saved_candidates = pickle.load(f)

    top_20 = saved_candidates['top_20']
    # Rank #5 is index 4
    z_rank5, rank5_traj, rank5_osc = top_20[4]
    print(f"  Rank #5 ultra-oscillatory: oscillation={rank5_osc:.4f}")

    # Convert z_rank5 to tensor
    z_rank5_tensor = torch.tensor(z_rank5, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    return stable_idx, z_rank5_tensor, stable_traj, rank5_traj


def encode_sample_to_latent(model, test_data, test_max_vals, idx):
    """Encode a single sample to latent space."""
    x_norm = torch.tensor(test_data[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    max_vals = torch.tensor(test_max_vals[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Encode
        x_rnn = x_norm.permute(0, 2, 1)
        _, (h_n, _) = model.encoder_rnn(x_rnn)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

        z_shape = model.shape_projector(encoded_summary)
        z_scale = model.scale_encoder(max_vals)
        bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
        mu = model.fc_mu(bottleneck_input)

    return mu, max_vals


def decode_latent_to_sample(model, z):
    """Decode latent vector to trajectory."""
    with torch.no_grad():
        # Use model's decode method
        recon, max_pred = model.decode(z)

    # Denormalize and apply extinction fix
    recon_np = recon.cpu().numpy()[0]
    max_pred_np = max_pred.cpu().numpy()[0]
    recon_denorm = recon_np * max_pred_np[:, np.newaxis]
    recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T

    return recon_fixed


def interpolate_latent_space(z_start, z_end, n_steps):
    """Linear interpolation in latent space."""
    alphas = np.linspace(0, 1, n_steps)
    interpolated = []

    for alpha in alphas:
        z_interp = (1 - alpha) * z_start + alpha * z_end
        interpolated.append(z_interp)

    return interpolated, alphas


def generate_interpolation_figure(stable_traj, osc_traj, interpolated_trajs, alphas):
    """Generate professional interpolation figure."""
    print("\nGenerating interpolation figure...")

    n_steps = len(interpolated_trajs)
    time_steps = np.arange(stable_traj.shape[1])

    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, n_steps, figure=fig, hspace=0.4, wspace=0.2,
                          height_ratios=[1, 0.8, 0.6])

    # ========================================================================
    # Row 0: Trajectory plots for each interpolation step
    # ========================================================================
    oscillation_strengths = []

    for step in range(n_steps):
        traj = interpolated_trajs[step]
        alpha = alphas[step]
        osc = compute_oscillation_strength(traj)
        oscillation_strengths.append(osc)

        # Time series
        ax1 = fig.add_subplot(gs[0, step])
        for sp in range(7):
            ax1.plot(time_steps, traj[sp, :], color=COLORS[sp],
                    linewidth=1.5, alpha=0.8)

        ax1.set_ylim([0, 1.2])
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([])  # Remove x-axis labels
        ax1.set_yticks([0, 0.5, 1.0])

        if step == 0:
            ax1.set_ylabel('Population', fontweight='bold', fontsize=9)
            title = 'Stable'
        elif step == n_steps - 1:
            title = 'Oscillatory'
        else:
            title = f'α={alpha:.2f}'

        ax1.set_title(title, fontweight='bold', fontsize=9, pad=6)

        if step == 0:
            ax1.legend([f'$x_{i}$' for i in range(7)], loc='upper right',
                      ncol=2, fontsize=6, framealpha=0.9)

    # ========================================================================
    # Row 1: Oscillation strength progression
    # ========================================================================
    ax_osc = fig.add_subplot(gs[1, :])

    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_steps))

    ax_osc.plot(alphas, oscillation_strengths, 'o-', color='black',
               linewidth=2.5, markersize=10, alpha=0.8)

    for i, (alpha, osc) in enumerate(zip(alphas, oscillation_strengths)):
        ax_osc.scatter(alpha, osc, color=colors_gradient[i], s=200,
                      edgecolor='black', linewidth=2, zorder=10)
        ax_osc.text(alpha, osc + 0.008, f'{osc:.3f}', ha='center',
                   va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.9, linewidth=0.5))

    ax_osc.set_xlabel('Interpolation Parameter (α)', fontweight='bold', fontsize=11)
    ax_osc.set_ylabel('Oscillation Strength', fontweight='bold', fontsize=11)
    ax_osc.set_title('Smooth Transition in Dynamical Characteristics',
                    fontweight='bold', fontsize=11, pad=10)
    ax_osc.grid(True, alpha=0.3)
    ax_osc.set_xlim([-0.05, 1.05])

    # ========================================================================
    # Row 2: Feature comparison
    # ========================================================================
    ax_features = fig.add_subplot(gs[2, :])

    # Compute features for each interpolation step
    mean_densities = [np.mean(traj) for traj in interpolated_trajs]
    final_densities = [np.mean(traj[:, -1]) for traj in interpolated_trajs]
    peak_densities = [np.max(traj) for traj in interpolated_trajs]

    # Normalize for comparison
    mean_densities_norm = (np.array(mean_densities) - min(mean_densities)) / (max(mean_densities) - min(mean_densities) + 1e-8)
    final_densities_norm = (np.array(final_densities) - min(final_densities)) / (max(final_densities) - min(final_densities) + 1e-8)
    peak_densities_norm = (np.array(peak_densities) - min(peak_densities)) / (max(peak_densities) - min(peak_densities) + 1e-8)
    osc_norm = (np.array(oscillation_strengths) - min(oscillation_strengths)) / (max(oscillation_strengths) - min(oscillation_strengths) + 1e-8)

    ax_features.plot(alphas, mean_densities_norm, 'o-', label='Mean Density',
                    linewidth=2, markersize=8, alpha=0.8)
    ax_features.plot(alphas, final_densities_norm, 's-', label='Final Density',
                    linewidth=2, markersize=8, alpha=0.8)
    ax_features.plot(alphas, peak_densities_norm, '^-', label='Peak Density',
                    linewidth=2, markersize=8, alpha=0.8)
    ax_features.plot(alphas, osc_norm, 'd-', label='Oscillation',
                    linewidth=2, markersize=8, alpha=0.8)

    ax_features.set_xlabel('Interpolation Parameter (α)', fontweight='bold', fontsize=10)
    ax_features.set_ylabel('Normalized Feature Value', fontweight='bold', fontsize=10)
    ax_features.set_title('Evolution of Dynamical Features During Interpolation',
                         fontweight='bold', fontsize=10, pad=8)
    ax_features.legend(loc='best', ncol=4, framealpha=0.95, fontsize=8)
    ax_features.grid(True, alpha=0.3)
    ax_features.set_xlim([-0.05, 1.05])
    ax_features.set_ylim([-0.05, 1.05])

    # Save
    output_png = f'{OUTPUT_DIR}fig_latent_interpolation.png'
    output_pdf = f'{OUTPUT_DIR}fig_latent_interpolation.pdf'

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"\n✓ Figure saved:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    plt.close()


def main():
    print("="*80)
    print("LATENT SPACE INTERPOLATION: STABLE → RANK #5 ULTRA-OSCILLATORY")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = LSTM_VAE(config=MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load test data
    print("Loading test data...")
    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)

    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    # Find extreme samples
    stable_idx, z_rank5, stable_traj, osc_traj = find_extreme_samples(
        test_data, test_max_vals, model, n_candidates=2000
    )

    # Encode stable sample to latent space
    print("\nEncoding stable sample to latent space...")
    z_stable, max_stable = encode_sample_to_latent(model, test_data, test_max_vals, stable_idx)
    z_osc = z_rank5  # Already in latent space from random sampling

    # Interpolate in latent space
    print(f"Performing {N_INTERPOLATION_STEPS}-step interpolation...")
    z_interpolated, alphas = interpolate_latent_space(
        z_stable, z_osc, N_INTERPOLATION_STEPS
    )

    # Decode interpolated latent codes
    print("Decoding interpolated latent codes...")
    interpolated_trajs = []

    for i, z in enumerate(tqdm(z_interpolated, desc="Decoding")):
        if i == len(z_interpolated) - 1:
            # Use the exact saved trajectory for the last step (α=1.0)
            interpolated_trajs.append(osc_traj)
        else:
            traj = decode_latent_to_sample(model, z)
            interpolated_trajs.append(traj)

    # Generate figure
    generate_interpolation_figure(stable_traj, osc_traj, interpolated_trajs, alphas)

    print("\n" + "="*80)
    print("✓ Interpolation figure generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
