"""
Visualize ultra-oscillatory dynamics from random latent space sampling.
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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
})

from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE
from utils.post_processing import apply_extinction_threshold

MODEL_CONFIG = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'
TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
COLORS = sns.color_palette("husl", 7)


def compute_oscillation_strength(trajectory):
    """Compute oscillation strength."""
    return np.mean([np.std(trajectory[sp, :]) for sp in range(trajectory.shape[0])])


def decode_latent_to_sample(model, z):
    """Decode latent vector to trajectory."""
    with torch.no_grad():
        recon, max_pred = model.decode(z)

    recon_np = recon.cpu().numpy()[0]
    max_pred_np = max_pred.cpu().numpy()[0]
    recon_denorm = recon_np * max_pred_np[:, np.newaxis]
    recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T

    return recon_fixed


def main():
    print("="*80)
    print("VISUALIZING ULTRA-OSCILLATORY DYNAMICS")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = LSTM_VAE(config=MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load test data to get latent space statistics
    print("Loading test data...")
    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)
    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    # Get latent space statistics (quick encoding of subset)
    print("Characterizing latent space...")
    test_latents = []
    with torch.no_grad():
        for i in tqdm(range(min(500, len(test_data))), desc="Encoding samples"):
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            x_rnn = x_norm.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(x_rnn)
            encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)
            z_shape = model.shape_projector(encoded_summary)
            z_scale = model.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
            mu = model.fc_mu(bottleneck_input)

            test_latents.append(mu.cpu().numpy()[0])

    test_latents = np.array(test_latents)
    latent_mean = test_latents.mean(axis=0)
    latent_std = test_latents.std(axis=0)

    print(f"  Latent mean: {latent_mean.mean():.3f}")
    print(f"  Latent std: {latent_std.mean():.3f}")

    # Sample many points and find ultra-oscillatory ones
    print("\nSearching for ultra-oscillatory samples (uniform sampling)...")
    n_samples = 10000
    candidates = []

    for i in tqdm(range(n_samples), desc="Sampling latent space"):
        # Uniform sampling in wider range
        z_random = np.random.uniform(
            latent_mean - 3*latent_std,
            latent_mean + 3*latent_std
        )
        z_tensor = torch.tensor(z_random, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        traj = decode_latent_to_sample(model, z_tensor)
        osc = compute_oscillation_strength(traj)

        candidates.append((z_random, traj, osc))

    # Sort by oscillation
    candidates.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop 10 oscillation strengths found:")
    for i in range(10):
        print(f"  {i+1}. Oscillation: {candidates[i][2]:.4f}")

    # Select diverse ultra-oscillatory samples
    top_samples = candidates[:20]  # Top 20

    # Create figure
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3,
                          height_ratios=[1, 1, 1, 1])

    time_steps = np.arange(65)

    for idx in range(20):
        z, traj, osc = top_samples[idx]

        row = idx // 5
        col = idx % 5

        ax = fig.add_subplot(gs[row, col])

        # Plot time series
        for sp in range(7):
            ax.plot(time_steps, traj[sp, :], color=COLORS[sp], linewidth=1.5, alpha=0.8)

        ax.set_ylim([0, 1.2])
        ax.set_xticks([0, 32, 64])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel('Population', fontsize=8)
        if row == 3:
            ax.set_xlabel('Time', fontsize=8)

        # Title with oscillation strength
        if idx == 0:
            title = f'#{idx+1}: {osc:.3f}\n(Maximum)'
        else:
            title = f'#{idx+1}: {osc:.3f}'

        ax.set_title(title, fontweight='bold', fontsize=9, pad=5)

        # Add background color gradient
        if idx == 0:
            ax.set_facecolor('#fff5f5')  # Lightest red tint for maximum

    # Add overall title
    fig.suptitle('Top 20 Ultra-Oscillatory Dynamics from Random Latent Space Sampling',
                fontweight='bold', fontsize=13, y=0.98)

    plt.savefig('final figures/fig_ultra_oscillatory_samples.png', dpi=300, bbox_inches='tight')
    plt.savefig('final figures/fig_ultra_oscillatory_samples.pdf', bbox_inches='tight')

    print(f"\n✓ Figure saved: final figures/fig_ultra_oscillatory_samples.png")

    # Save the top candidates for reuse
    print("\nSaving top ultra-oscillatory candidates...")
    candidates_to_save = {
        'top_20': top_samples,  # List of (z, traj, osc)
        'latent_mean': latent_mean,
        'latent_std': latent_std
    }
    with open('data/ultra_oscillatory_candidates.pkl', 'wb') as f:
        pickle.dump(candidates_to_save, f)
    print("✓ Saved to data/ultra_oscillatory_candidates.pkl")

    # Also create a detailed figure of top 6
    print("\nGenerating detailed figure of top 6...")

    fig2 = plt.figure(figsize=(15, 10))
    gs2 = gridspec.GridSpec(3, 6, figure=fig2, hspace=0.4, wspace=0.3,
                           height_ratios=[1, 1, 0.8])

    for idx in range(6):
        z, traj, osc = top_samples[idx]

        # Time series
        ax1 = fig2.add_subplot(gs2[0, idx])
        for sp in range(7):
            ax1.plot(time_steps, traj[sp, :], color=COLORS[sp], linewidth=2, alpha=0.9)
        ax1.set_ylim([0, 1.2])
        ax1.set_xticks([])
        ax1.set_yticks([0, 0.5, 1.0])
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Rank #{idx+1}: Osc={osc:.4f}', fontweight='bold', fontsize=10)

        if idx == 0:
            ax1.set_ylabel('Population', fontweight='bold')
            ax1.legend([f'$x_{i}$' for i in range(7)], loc='upper right',
                      ncol=2, fontsize=7, framealpha=0.9)

        # Phase portrait (x0 vs x1)
        ax2 = fig2.add_subplot(gs2[1, idx])
        ax2.plot(traj[0, :], traj[1, :], color='steelblue', linewidth=2, alpha=0.8)
        ax2.scatter(traj[0, 0], traj[1, 0], color='green', s=50, marker='o', zorder=10)
        ax2.scatter(traj[0, -1], traj[1, -1], color='red', s=50, marker='s', zorder=10)
        ax2.set_xlim([0, 1.2])
        ax2.set_ylim([0, 1.2])
        ax2.set_xticks([0, 0.5, 1.0])
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('$x_0$', fontsize=8)

        if idx == 0:
            ax2.set_ylabel('$x_1$', fontweight='bold')

        # Heatmap of all species
        ax3 = fig2.add_subplot(gs2[2, idx])
        im = ax3.imshow(traj, aspect='auto', cmap='viridis', interpolation='nearest',
                       vmin=0, vmax=1.0)
        ax3.set_xlabel('Time', fontsize=8)
        ax3.set_yticks(range(7))
        ax3.set_yticklabels([f'$x_{i}$' for i in range(7)], fontsize=7)

        if idx == 0:
            ax3.set_ylabel('Species', fontweight='bold', fontsize=9)

    fig2.suptitle('Detailed View: Top 6 Ultra-Oscillatory Dynamics',
                 fontweight='bold', fontsize=13, y=0.98)

    plt.savefig('final figures/fig_ultra_oscillatory_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig('final figures/fig_ultra_oscillatory_detailed.pdf', bbox_inches='tight')

    print(f"✓ Figure saved: final figures/fig_ultra_oscillatory_detailed.png")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
