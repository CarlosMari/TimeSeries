"""
Exploration: Can we generate ULTRA-oscillatory dynamics by extrapolating in latent space?

Strategy:
1. Find multiple highly oscillatory samples
2. Identify the "oscillation direction" in latent space
3. Extrapolate beyond to generate synthetic extreme oscillations
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

TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'
COLORS = sns.color_palette("husl", 7)


def compute_oscillation_strength(trajectory):
    """Compute oscillation strength."""
    return np.mean([np.std(trajectory[sp, :]) for sp in range(trajectory.shape[0])])


def encode_sample_to_latent(model, test_data, test_max_vals, idx):
    """Encode a single sample to latent space."""
    x_norm = torch.tensor(test_data[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    max_vals = torch.tensor(test_max_vals[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x_rnn = x_norm.permute(0, 2, 1)
        _, (h_n, _) = model.encoder_rnn(x_rnn)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)
        z_shape = model.shape_projector(encoded_summary)
        z_scale = model.scale_encoder(max_vals)
        bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
        mu = model.fc_mu(bottleneck_input)

    return mu


def decode_latent_to_sample(model, z):
    """Decode latent vector to trajectory."""
    with torch.no_grad():
        recon, max_pred = model.decode(z)

    recon_np = recon.cpu().numpy()[0]
    max_pred_np = max_pred.cpu().numpy()[0]
    recon_denorm = recon_np * max_pred_np[:, np.newaxis]
    recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T

    return recon_fixed


def find_oscillatory_samples(test_data, test_max_vals, model, n_candidates=2000, top_k=10):
    """Find top oscillatory samples."""
    print(f"\nFinding top {top_k} oscillatory samples from {n_candidates} candidates...")

    oscillation_scores = []

    with torch.no_grad():
        for i in tqdm(range(min(n_candidates, len(test_data))), desc="Computing oscillations"):
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            recon, _, _, max_pred, _ = model(x_norm, max_vals, teacher_forcing_ratio=0.0)
            recon_denorm = recon.cpu().numpy()[0] * max_pred.cpu().numpy()[0][:, np.newaxis]
            recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T

            osc = compute_oscillation_strength(recon_fixed)
            oscillation_scores.append((i, osc))

    oscillation_scores.sort(key=lambda x: x[1], reverse=True)
    top_oscillatory = oscillation_scores[:top_k]

    print(f"\nTop {top_k} oscillatory samples:")
    for idx, osc in top_oscillatory:
        print(f"  idx={idx}, oscillation={osc:.4f}")

    return [idx for idx, _ in top_oscillatory]


def main():
    print("="*80)
    print("EXPLORING OSCILLATION EXTRAPOLATION IN LATENT SPACE")
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

    # Get latent space statistics for sampling
    print("Characterizing latent space...")
    test_latents = []
    with torch.no_grad():
        for i in tqdm(range(min(500, len(test_data))), desc="Encoding samples"):
            z = encode_sample_to_latent(model, test_data, test_max_vals, i)
            test_latents.append(z.cpu().numpy()[0])
    test_latents = np.array(test_latents)
    latent_mean = test_latents.mean(axis=0)
    latent_std = test_latents.std(axis=0)

    # Sample latent space to find ultra-oscillatory samples
    print("\nSearching for ultra-oscillatory samples (uniform sampling)...")
    n_samples = 10000
    candidates = []

    for i in tqdm(range(n_samples), desc="Sampling latent space"):
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

    # Use Rank 5 (index 4) as reference oscillatory point
    z_rank5, _, osc_rank5 = candidates[4]
    print(f"\nUsing Rank #5 as reference: oscillation={osc_rank5:.4f}")

    # Find a stable sample
    print("\nFinding a stable sample...")
    oscillation_all = []
    with torch.no_grad():
        for i in tqdm(range(min(1000, len(test_data))), desc="Finding stable"):
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            recon, _, _, max_pred, _ = model(x_norm, max_vals, teacher_forcing_ratio=0.0)
            recon_denorm = recon.cpu().numpy()[0] * max_pred.cpu().numpy()[0][:, np.newaxis]
            recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T
            osc = compute_oscillation_strength(recon_fixed)
            oscillation_all.append((i, osc))

    oscillation_all.sort(key=lambda x: x[1])
    stable_idx = oscillation_all[0][0]
    stable_osc = oscillation_all[0][1]
    print(f"Stable sample: idx={stable_idx}, oscillation={stable_osc:.4f}")

    z_stable = encode_sample_to_latent(model, test_data, test_max_vals, stable_idx).cpu().numpy()[0]

    # Compute oscillation direction (from stable to Rank 5)
    oscillation_direction = z_rank5 - z_stable
    direction_magnitude = np.linalg.norm(oscillation_direction)

    print(f"\nOscillation direction computed (from stable to Rank #5)")
    print(f"Direction magnitude: {direction_magnitude:.3f}")

    # Now extrapolate: α=0 (stable), α=1 (Rank #5), α>1 (extrapolation)
    print("\nExtrapolating from stable to Rank #5 and beyond...")
    extrapolation_factors = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # α=1 is Rank #5

    trajectories = []
    oscillations = []

    for alpha in tqdm(extrapolation_factors, desc="Extrapolating"):
        # Interpolate/extrapolate from stable to Rank #5
        z_extrap = z_stable + alpha * oscillation_direction
        z_extrap_tensor = torch.tensor(z_extrap, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Decode
        traj = decode_latent_to_sample(model, z_extrap_tensor)
        osc = compute_oscillation_strength(traj)

        trajectories.append(traj)
        oscillations.append(osc)

        print(f"  α={alpha:.1f}: oscillation={osc:.4f}")

    # Visualize results
    print("\nGenerating visualization...")
    n_steps = len(extrapolation_factors)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, n_steps, figure=fig, hspace=0.4, wspace=0.2,
                          height_ratios=[1, 0.6, 0.6])

    time_steps = np.arange(trajectories[0].shape[1])

    # Row 0: Time series
    for step in range(n_steps):
        ax = fig.add_subplot(gs[0, step])
        traj = trajectories[step]

        for sp in range(7):
            ax.plot(time_steps, traj[sp, :], color=COLORS[sp], linewidth=1.5, alpha=0.8)

        ax.set_ylim([0, 1.2])
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(True, alpha=0.3)

        if step == 0:
            ax.set_ylabel('Population', fontweight='bold', fontsize=9)
            title = 'Stable'
        elif extrapolation_factors[step] == 1.0:
            title = 'Rank #5'
        elif extrapolation_factors[step] > 1.0:
            title = f'α={extrapolation_factors[step]:.1f}\n(Extrap.)'
        else:
            title = f'α={extrapolation_factors[step]:.1f}'

        ax.set_title(title, fontweight='bold', fontsize=9, pad=6)

    # Row 1: Oscillation progression
    ax_osc = fig.add_subplot(gs[1, :])

    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, n_steps))

    ax_osc.plot(extrapolation_factors, oscillations, 'o-', color='black',
               linewidth=2.5, markersize=10, alpha=0.8)

    for i, (alpha, osc) in enumerate(zip(extrapolation_factors, oscillations)):
        ax_osc.scatter(alpha, osc, color=colors_gradient[i], s=200,
                      edgecolor='black', linewidth=2, zorder=10)
        ax_osc.text(alpha, osc + 0.01, f'{osc:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.9, linewidth=0.5))

    # Mark extrapolation region
    ax_osc.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_osc.text(1.0, ax_osc.get_ylim()[1]*0.95, 'Rank #5\nReference',
               ha='center', va='top', fontsize=8, color='red',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    ax_osc.axvspan(1.0, max(extrapolation_factors), alpha=0.15, color='red',
                   label='Extrapolation Region')

    ax_osc.set_xlabel('Extrapolation Factor (α)', fontweight='bold', fontsize=11)
    ax_osc.set_ylabel('Oscillation Strength', fontweight='bold', fontsize=11)
    ax_osc.set_title('Extrapolating from Stable to Ultra-Oscillatory (Rank #5)', fontweight='bold',
                    fontsize=11, pad=10)
    ax_osc.legend(loc='upper left', fontsize=9)
    ax_osc.grid(True, alpha=0.3)

    # Row 2: Phase portraits
    for step in range(n_steps):
        ax = fig.add_subplot(gs[2, step])
        traj = trajectories[step]

        ax.plot(traj[0, :], traj[1, :], color='steelblue', linewidth=1.5, alpha=0.8)
        ax.scatter(traj[0, 0], traj[1, 0], color='green', s=40, marker='o', zorder=10)
        ax.scatter(traj[0, -1], traj[1, -1], color='red', s=40, marker='s', zorder=10)

        ax.set_xlim([0, 1.2])
        ax.set_ylim([0, 1.2])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('$x_0$', fontsize=8)

        if step == 0:
            ax.set_ylabel('$x_1$', fontweight='bold', fontsize=9)

    plt.savefig('final figures/fig_oscillation_extrapolation.png', dpi=300, bbox_inches='tight')
    plt.savefig('final figures/fig_oscillation_extrapolation.pdf', bbox_inches='tight')

    print("\n✓ Figure saved: final figures/fig_oscillation_extrapolation.png")

    # Key finding
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    rank5_oscillation = oscillations[extrapolation_factors.index(1.0)]
    max_generated = max(oscillations)
    stable_oscillation = oscillations[0]
    improvement = ((max_generated - rank5_oscillation) / rank5_oscillation) * 100

    print(f"Stable oscillation (α=0): {stable_oscillation:.4f}")
    print(f"Rank #5 oscillation (α=1): {rank5_oscillation:.4f}")
    print(f"Maximum generated oscillation (extrapolated): {max_generated:.4f}")
    print(f"Improvement beyond Rank #5: {improvement:+.1f}%")

    if max_generated > rank5_oscillation * 1.1:
        print("\n✓ SUCCESS: Extrapolation INCREASES oscillations beyond Rank #5!")
        print("  The latent space extends the oscillation axis")
    else:
        print("\n✗ Extrapolation saturates near Rank #5")
        print("  The model may have learned bounds on valid dynamics")

    print("="*80)


if __name__ == "__main__":
    main()
