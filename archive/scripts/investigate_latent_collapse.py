"""
Investigate potential posterior collapse in the scale-conditioned CVAE.

Check if latent dimensions are actually being used or if some have collapsed to zero variance.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE

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

TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
MODEL_PATH = 'model_ckpts/model_final_50_conditioned.pth'


def investigate_latent_space():
    """Check latent space statistics to detect collapse."""
    print("="*80)
    print("INVESTIGATING LATENT SPACE FOR POSTERIOR COLLAPSE")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = LSTM_VAE(config=model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load test data
    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)
    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    print(f"Test samples: {len(test_data)}")

    # Sample subset for analysis
    n_samples = min(5000, len(test_data))
    print(f"Analyzing {n_samples} samples...")

    mus = []
    log_vars = []
    zs = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Encoding"):
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Forward pass
            x_rnn = x_norm.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(x_rnn)
            encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

            # Get latent representation
            z_shape = model.shape_projector(encoded_summary)
            z_scale = model.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)

            mu = model.fc_mu(bottleneck_input)
            log_var = model.fc_log_var(bottleneck_input)

            # Sample z
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std

            mus.append(mu.cpu().numpy()[0])
            log_vars.append(log_var.cpu().numpy()[0])
            zs.append(z.cpu().numpy()[0])

    mus = np.array(mus)
    log_vars = np.array(log_vars)
    zs = np.array(zs)

    # Analyze statistics
    print("\n" + "="*80)
    print("LATENT SPACE STATISTICS")
    print("="*80)

    # Variance of each dimension
    mu_vars = np.var(mus, axis=0)
    z_vars = np.var(zs, axis=0)

    # Mean of log_vars (indicates how much the model is using each dimension)
    mean_log_vars = np.mean(log_vars, axis=0)
    mean_vars = np.exp(mean_log_vars)

    print(f"\nMu statistics (mean of posterior):")
    print(f"  Dimensions with near-zero variance (< 0.01): {np.sum(mu_vars < 0.01)}/{len(mu_vars)}")
    print(f"  Dimensions with near-zero variance (< 0.001): {np.sum(mu_vars < 0.001)}/{len(mu_vars)}")
    print(f"  Min variance: {mu_vars.min():.6f}")
    print(f"  Max variance: {mu_vars.max():.6f}")
    print(f"  Mean variance: {mu_vars.mean():.6f}")

    print(f"\nZ statistics (sampled latent codes):")
    print(f"  Dimensions with near-zero variance (< 0.01): {np.sum(z_vars < 0.01)}/{len(z_vars)}")
    print(f"  Dimensions with near-zero variance (< 0.001): {np.sum(z_vars < 0.001)}/{len(z_vars)}")
    print(f"  Min variance: {z_vars.min():.6f}")
    print(f"  Max variance: {z_vars.max():.6f}")
    print(f"  Mean variance: {z_vars.mean():.6f}")

    print(f"\nPosterior variance (exp(log_var)):")
    print(f"  Mean of mean_vars: {mean_vars.mean():.6f}")
    print(f"  Dimensions with variance close to 1.0 (0.9-1.1): {np.sum((mean_vars > 0.9) & (mean_vars < 1.1))}/{len(mean_vars)}")

    # Find collapsed dimensions
    threshold = 0.01
    collapsed_dims = np.where(mu_vars < threshold)[0]
    active_dims = np.where(mu_vars >= threshold)[0]

    print(f"\nCOLLAPSED DIMENSIONS (variance < {threshold}):")
    print(f"  Count: {len(collapsed_dims)}/{len(mu_vars)}")
    print(f"  Indices: {collapsed_dims.tolist()}")

    print(f"\nACTIVE DIMENSIONS (variance >= {threshold}):")
    print(f"  Count: {len(active_dims)}/{len(mu_vars)}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Mu variance per dimension
    ax = axes[0, 0]
    ax.bar(range(len(mu_vars)), mu_vars, color='steelblue', alpha=0.7)
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax.set_xlabel('Latent Dimension', fontweight='bold')
    ax.set_ylabel('Variance of μ', fontweight='bold')
    ax.set_title('Variance of Posterior Mean (μ) Across Dimensions', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Z variance per dimension
    ax = axes[0, 1]
    ax.bar(range(len(z_vars)), z_vars, color='orange', alpha=0.7)
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax.set_xlabel('Latent Dimension', fontweight='bold')
    ax.set_ylabel('Variance of z', fontweight='bold')
    ax.set_title('Variance of Sampled Latent Codes (z) Across Dimensions', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Mean posterior variance per dimension
    ax = axes[1, 0]
    ax.bar(range(len(mean_vars)), mean_vars, color='green', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Prior variance (1.0)')
    ax.set_xlabel('Latent Dimension', fontweight='bold')
    ax.set_ylabel('Mean Posterior Variance', fontweight='bold')
    ax.set_title('Mean Posterior Variance (σ²) Across Dimensions', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Distribution of mu values for top 5 and bottom 5 dimensions
    ax = axes[1, 1]
    top_dims = np.argsort(mu_vars)[-5:]
    bottom_dims = np.argsort(mu_vars)[:5]

    positions = []
    data_to_plot = []
    labels = []
    colors = []

    for i, dim in enumerate(top_dims):
        positions.append(i)
        data_to_plot.append(mus[:, dim])
        labels.append(f'Dim {dim}\n(active)')
        colors.append('steelblue')

    for i, dim in enumerate(bottom_dims):
        positions.append(len(top_dims) + i)
        data_to_plot.append(mus[:, dim])
        labels.append(f'Dim {dim}\n(collapsed)')
        colors.append('red')

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('μ values', fontweight='bold')
    ax.set_title('Distribution of μ: Top 5 vs Bottom 5 Dimensions', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('latent_collapse_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: latent_collapse_analysis.png")
    plt.close()

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    if len(collapsed_dims) > len(mu_vars) * 0.3:
        print(f"⚠️  WARNING: Significant posterior collapse detected!")
        print(f"   {len(collapsed_dims)} out of {len(mu_vars)} dimensions ({len(collapsed_dims)/len(mu_vars)*100:.1f}%) are collapsed.")
        print(f"\nPossible causes:")
        print(f"  1. KL weight (beta) may be too high during training")
        print(f"  2. Latent dimension (50) may be too large for the task")
        print(f"  3. Scale conditioning reduces information needed in latent space")
        print(f"\nRecommendations:")
        print(f"  - Consider reducing latent_dim to {len(active_dims)} dimensions")
        print(f"  - Or retrain with lower beta_max or longer warmup")
    else:
        print(f"✓ Latent space is healthy.")
        print(f"  Only {len(collapsed_dims)} dimensions collapsed ({len(collapsed_dims)/len(mu_vars)*100:.1f}%).")

    print("="*80)

    return mu_vars, z_vars, mean_vars, collapsed_dims, active_dims


if __name__ == "__main__":
    investigate_latent_space()
