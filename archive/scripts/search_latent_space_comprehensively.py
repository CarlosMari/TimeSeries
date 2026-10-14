"""
Comprehensive latent space search: Are there hidden regions with extreme oscillations?

Strategy:
1. Randomly sample many points across the latent space
2. Decode and measure oscillation strength
3. Find if there are undiscovered high-oscillation regions
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
N_RANDOM_SAMPLES = 5000  # Sample 5000 random points


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


def main():
    print("="*80)
    print("COMPREHENSIVE LATENT SPACE SEARCH")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = LSTM_VAE(config=MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load test data to get latent space statistics
    print("Loading test data to characterize latent space...")
    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)
    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    # Encode a subset of test samples to get latent space statistics
    print("Encoding 2000 test samples to estimate latent distribution...")
    test_latents = []
    test_oscillations = []

    with torch.no_grad():
        for i in tqdm(range(min(2000, len(test_data))), desc="Encoding test set"):
            z = encode_sample_to_latent(model, test_data, test_max_vals, i)
            test_latents.append(z.cpu().numpy()[0])

            # Also compute oscillation for test sample
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            recon, _, _, max_pred, _ = model(x_norm, max_vals, teacher_forcing_ratio=0.0)
            recon_denorm = recon.cpu().numpy()[0] * max_pred.cpu().numpy()[0][:, np.newaxis]
            recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T
            osc = compute_oscillation_strength(recon_fixed)
            test_oscillations.append(osc)

    test_latents = np.array(test_latents)
    test_oscillations = np.array(test_oscillations)

    # Get latent space statistics
    latent_mean = test_latents.mean(axis=0)
    latent_std = test_latents.std(axis=0)

    print(f"\nLatent space statistics:")
    print(f"  Mean: {latent_mean.mean():.3f} ± {latent_mean.std():.3f}")
    print(f"  Std: {latent_std.mean():.3f} ± {latent_std.std():.3f}")
    print(f"\nTest set oscillations: {test_oscillations.min():.4f} to {test_oscillations.max():.4f}")
    print(f"  Mean: {test_oscillations.mean():.4f}, Std: {test_oscillations.std():.4f}")

    # Strategy 1: Random sampling from learned distribution (Gaussian)
    print(f"\n{'='*80}")
    print(f"STRATEGY 1: Random sampling from Gaussian N(μ, σ²) distribution")
    print(f"{'='*80}")

    random_oscillations_gaussian = []

    for i in tqdm(range(N_RANDOM_SAMPLES), desc="Gaussian sampling"):
        # Sample from Gaussian centered at latent mean with observed std
        z_random = np.random.normal(latent_mean, latent_std)
        z_tensor = torch.tensor(z_random, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        traj = decode_latent_to_sample(model, z_tensor)
        osc = compute_oscillation_strength(traj)
        random_oscillations_gaussian.append(osc)

    random_oscillations_gaussian = np.array(random_oscillations_gaussian)

    print(f"\nGaussian sampling results:")
    print(f"  Range: {random_oscillations_gaussian.min():.4f} to {random_oscillations_gaussian.max():.4f}")
    print(f"  Mean: {random_oscillations_gaussian.mean():.4f}")
    print(f"  95th percentile: {np.percentile(random_oscillations_gaussian, 95):.4f}")
    print(f"  99th percentile: {np.percentile(random_oscillations_gaussian, 99):.4f}")

    # Strategy 2: Uniform sampling from wider range
    print(f"\n{'='*80}")
    print(f"STRATEGY 2: Uniform sampling from wider range [-3σ, +3σ]")
    print(f"{'='*80}")

    random_oscillations_uniform = []

    for i in tqdm(range(N_RANDOM_SAMPLES), desc="Uniform sampling"):
        # Sample uniformly in a wider range
        z_random = np.random.uniform(
            latent_mean - 3*latent_std,
            latent_mean + 3*latent_std
        )
        z_tensor = torch.tensor(z_random, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        traj = decode_latent_to_sample(model, z_tensor)
        osc = compute_oscillation_strength(traj)
        random_oscillations_uniform.append(osc)

    random_oscillations_uniform = np.array(random_oscillations_uniform)

    print(f"\nUniform sampling results:")
    print(f"  Range: {random_oscillations_uniform.min():.4f} to {random_oscillations_uniform.max():.4f}")
    print(f"  Mean: {random_oscillations_uniform.mean():.4f}")
    print(f"  95th percentile: {np.percentile(random_oscillations_uniform, 95):.4f}")
    print(f"  99th percentile: {np.percentile(random_oscillations_uniform, 99):.4f}")

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")

    max_test = test_oscillations.max()
    max_gaussian = random_oscillations_gaussian.max()
    max_uniform = random_oscillations_uniform.max()

    print(f"\nMaximum oscillation found:")
    print(f"  Test set (observed):        {max_test:.4f}")
    print(f"  Gaussian random sampling:   {max_gaussian:.4f}  ({(max_gaussian/max_test-1)*100:+.1f}%)")
    print(f"  Uniform random sampling:    {max_uniform:.4f}  ({(max_uniform/max_test-1)*100:+.1f}%)")

    if max(max_gaussian, max_uniform) > max_test * 1.05:
        print(f"\n✓ FOUND HIGHER OSCILLATIONS through random sampling!")
        print(f"  The test set doesn't fully explore the latent space")
    else:
        print(f"\n✗ No higher oscillations found")
        print(f"  The test set appears to cover the high-oscillation regions well")

    # Visualize distributions
    print(f"\nGenerating comparison figure...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram comparison
    ax = axes[0, 0]
    ax.hist(test_oscillations, bins=50, alpha=0.6, label='Test Set', density=True)
    ax.hist(random_oscillations_gaussian, bins=50, alpha=0.6, label='Gaussian Sampling', density=True)
    ax.hist(random_oscillations_uniform, bins=50, alpha=0.6, label='Uniform Sampling', density=True)
    ax.set_xlabel('Oscillation Strength')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(alpha=0.3)

    # CDF comparison
    ax = axes[0, 1]
    test_sorted = np.sort(test_oscillations)
    gaussian_sorted = np.sort(random_oscillations_gaussian)
    uniform_sorted = np.sort(random_oscillations_uniform)

    ax.plot(test_sorted, np.linspace(0, 1, len(test_sorted)),
            label='Test Set', linewidth=2)
    ax.plot(gaussian_sorted, np.linspace(0, 1, len(gaussian_sorted)),
            label='Gaussian Sampling', linewidth=2)
    ax.plot(uniform_sorted, np.linspace(0, 1, len(uniform_sorted)),
            label='Uniform Sampling', linewidth=2)
    ax.set_xlabel('Oscillation Strength')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot comparison
    ax = axes[1, 0]
    data_to_plot = [test_oscillations, random_oscillations_gaussian, random_oscillations_uniform]
    bp = ax.boxplot(data_to_plot, labels=['Test Set', 'Gaussian\nSampling', 'Uniform\nSampling'],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax.set_ylabel('Oscillation Strength')
    ax.set_title('Distribution Statistics')
    ax.grid(axis='y', alpha=0.3)

    # Tail comparison (focus on high oscillations)
    ax = axes[1, 1]
    ax.hist(test_oscillations[test_oscillations > 0.15], bins=30, alpha=0.6,
            label=f'Test (n={np.sum(test_oscillations > 0.15)})', density=True)
    ax.hist(random_oscillations_gaussian[random_oscillations_gaussian > 0.15], bins=30, alpha=0.6,
            label=f'Gaussian (n={np.sum(random_oscillations_gaussian > 0.15)})', density=True)
    ax.hist(random_oscillations_uniform[random_oscillations_uniform > 0.15], bins=30, alpha=0.6,
            label=f'Uniform (n={np.sum(random_oscillations_uniform > 0.15)})', density=True)
    ax.set_xlabel('Oscillation Strength')
    ax.set_ylabel('Density')
    ax.set_title('High Oscillation Tail (>0.15)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('final figures/fig_latent_space_comprehensive_search.png', dpi=300, bbox_inches='tight')
    plt.savefig('final figures/fig_latent_space_comprehensive_search.pdf', bbox_inches='tight')

    print(f"✓ Figure saved: final figures/fig_latent_space_comprehensive_search.png")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
