"""
Latent Space Family Analysis: Do different regions encode different dynamics?

This script:
1. Encodes all test samples to latent space
2. Clusters the latent representations
3. Analyzes characteristics of each cluster (family)
4. Creates professional figure for Chaos, Solitons and Fractals

Professional figure showing that different latent regions → different ecological dynamics
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP
from scipy.stats import zscore
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
    "latent_dim": 30,  # Use 30D model for cleaner analysis
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'
OUTPUT_DIR = 'final figures/'

N_CLUSTERS = 5  # Number of families to identify
N_SAMPLES_TO_ENCODE = 5000  # Subsample for faster analysis
N_EXAMPLES_PER_CLUSTER = 6  # Examples to show per family

COLORS_FAMILIES = sns.color_palette("Set2", N_CLUSTERS)
COLORS_SPECIES = sns.color_palette("husl", 7)


def encode_test_set_to_latent(model, test_data, test_max_vals, n_samples):
    """Encode test samples to latent space (μ vectors)."""
    print(f"\nEncoding {n_samples} samples to latent space...")

    latent_codes = np.zeros((n_samples, MODEL_CONFIG['latent_dim']))
    reconstructions = []

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Encoding"):
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Encode to latent space
            x_rnn = x_norm.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(x_rnn)
            encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

            z_shape = model.shape_projector(encoded_summary)
            z_scale = model.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)
            mu = model.fc_mu(bottleneck_input)

            latent_codes[i] = mu.cpu().numpy()[0]

            # Also get reconstruction for analysis
            recon, _, _, max_pred, _ = model(x_norm, max_vals, teacher_forcing_ratio=0.0)
            recon_denorm = recon.cpu().numpy()[0] * max_pred.cpu().numpy()[0][:, np.newaxis]
            recon_fixed = apply_extinction_threshold(recon_denorm.T, threshold=0.005).T
            reconstructions.append(recon_fixed)

    return latent_codes, np.array(reconstructions)


def cluster_latent_space(latent_codes, n_clusters):
    """Cluster latent representations using K-means."""
    print(f"\nClustering latent space into {n_clusters} families...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(latent_codes)

    print(f"  Cluster sizes: {np.bincount(labels)}")

    return labels, kmeans


def compute_trajectory_features(trajectories):
    """
    Compute interpretable features for each trajectory.

    Returns dict with:
    - species_richness: number of surviving species at end
    - mean_density: average population density
    - oscillation_strength: measure of temporal variation
    - extinction_time: mean time to extinction (if any)
    """
    n_samples = len(trajectories)
    features = {
        'species_richness': np.zeros(n_samples),
        'mean_density': np.zeros(n_samples),
        'oscillation_strength': np.zeros(n_samples),
        'final_density': np.zeros(n_samples),
        'peak_density': np.zeros(n_samples),
    }

    for i, traj in enumerate(trajectories):
        # Species richness: count species with final density > 0.01
        final_state = traj[:, -1]
        features['species_richness'][i] = np.sum(final_state > 0.01)

        # Mean density across all species and time
        features['mean_density'][i] = np.mean(traj)

        # Oscillation strength: mean std deviation across time for each species
        oscillations = np.mean([np.std(traj[sp, :]) for sp in range(7)])
        features['oscillation_strength'][i] = oscillations

        # Final density (mean across species)
        features['final_density'][i] = np.mean(final_state)

        # Peak density
        features['peak_density'][i] = np.max(traj)

    return features


def generate_latent_family_figure(latent_codes, labels, reconstructions, kmeans):
    """Generate comprehensive figure showing latent space families."""
    print("\nGenerating latent space families figure...")

    # Compute features for all trajectories
    features = compute_trajectory_features(reconstructions)

    # PCA projection for visualization
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_codes)

    # Create figure - simpler layout
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 5, figure=fig,
                          hspace=0.35, wspace=0.3)

    # ========================================================================
    # Panel A: Latent Space Clustering (PCA projection) - spans 2 columns
    # ========================================================================
    ax_latent = fig.add_subplot(gs[0:2, 0:2])  # Span rows 0-1, cols 0-1

    for cluster_id in range(N_CLUSTERS):
        mask = labels == cluster_id
        ax_latent.scatter(latent_pca[mask, 0], latent_pca[mask, 1],
                         c=[COLORS_FAMILIES[cluster_id]], alpha=0.6, s=15,
                         label=f'Family {cluster_id+1} (n={np.sum(mask)})',
                         edgecolors='none')

    # Plot cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax_latent.scatter(centers_pca[:, 0], centers_pca[:, 1],
                     c='black', marker='X', s=200, edgecolors='white',
                     linewidths=2, zorder=10)

    ax_latent.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax_latent.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax_latent.set_title('A) Latent Space Clustering (PCA Projection)',
                       fontweight='bold', loc='left', pad=10)
    ax_latent.legend(loc='best', framealpha=0.95, fontsize=7)
    ax_latent.grid(True, alpha=0.3)

    # ========================================================================
    # Panels B-F: Representative samples from each family
    # ========================================================================
    time_steps = np.arange(reconstructions.shape[2])

    for cluster_id in range(N_CLUSTERS):
        # Get samples from this cluster
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]

        # Select representative samples (closest to cluster center)
        cluster_latents = latent_codes[mask]
        distances = np.linalg.norm(cluster_latents - kmeans.cluster_centers_[cluster_id], axis=1)
        representative_indices = cluster_indices[np.argsort(distances)[:N_EXAMPLES_PER_CLUSTER]]

        # Determine position: first 3 families in row 0 (cols 2-4), last 2 in row 1 (cols 2-3)
        if cluster_id < 3:
            row = 0
            col = cluster_id + 2  # cols 2, 3, 4
        else:
            row = 1
            col = cluster_id - 1  # cluster 3->col 2, cluster 4->col 3

        ax = fig.add_subplot(gs[row, col])

        # Show all examples overlaid
        for idx in representative_indices:
            for sp in range(7):
                ax.plot(time_steps, reconstructions[idx, sp, :],
                        color=COLORS_SPECIES[sp], alpha=0.4, linewidth=0.8)

        ax.set_xlabel('Time step', fontsize=8)
        ax.set_ylabel('Population', fontsize=8)
        ax.set_title(f'{chr(66+cluster_id)}) Family {cluster_id+1} (n={np.sum(mask)})',
                     fontweight='bold', loc='left', fontsize=9, pad=5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, None])

    # ========================================================================
    # Bottom row: Statistical comparison of families
    # ========================================================================

    # Species richness comparison
    ax_rich = fig.add_subplot(gs[2, 0])
    richness_by_cluster = [features['species_richness'][labels == i] for i in range(N_CLUSTERS)]
    bp1 = ax_rich.boxplot(richness_by_cluster, patch_artist=True, widths=0.6,
                          boxprops=dict(linewidth=1.0),
                          whiskerprops=dict(linewidth=1.0),
                          capprops=dict(linewidth=1.0),
                          medianprops=dict(linewidth=1.5, color='red'))

    for patch, color in zip(bp1['boxes'], COLORS_FAMILIES):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_rich.set_xlabel('Family', fontweight='bold')
    ax_rich.set_ylabel('Species Richness', fontweight='bold')
    ax_rich.set_title('Species Diversity', fontweight='bold', fontsize=9)
    ax_rich.grid(axis='y', alpha=0.3)
    ax_rich.set_xticklabels([f'F{i+1}' for i in range(N_CLUSTERS)])

    # Mean density comparison
    ax_dens = fig.add_subplot(gs[2, 1])
    density_by_cluster = [features['mean_density'][labels == i] for i in range(N_CLUSTERS)]
    bp2 = ax_dens.boxplot(density_by_cluster, patch_artist=True, widths=0.6,
                          boxprops=dict(linewidth=1.0),
                          whiskerprops=dict(linewidth=1.0),
                          capprops=dict(linewidth=1.0),
                          medianprops=dict(linewidth=1.5, color='red'))

    for patch, color in zip(bp2['boxes'], COLORS_FAMILIES):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_dens.set_xlabel('Family', fontweight='bold')
    ax_dens.set_ylabel('Mean Density', fontweight='bold')
    ax_dens.set_title('Population Density', fontweight='bold', fontsize=9)
    ax_dens.grid(axis='y', alpha=0.3)
    ax_dens.set_xticklabels([f'F{i+1}' for i in range(N_CLUSTERS)])

    # Oscillation strength
    ax_osc = fig.add_subplot(gs[2, 2])
    osc_by_cluster = [features['oscillation_strength'][labels == i] for i in range(N_CLUSTERS)]
    bp3 = ax_osc.boxplot(osc_by_cluster, patch_artist=True, widths=0.6,
                         boxprops=dict(linewidth=1.0),
                         whiskerprops=dict(linewidth=1.0),
                         capprops=dict(linewidth=1.0),
                         medianprops=dict(linewidth=1.5, color='red'))

    for patch, color in zip(bp3['boxes'], COLORS_FAMILIES):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_osc.set_xlabel('Family', fontweight='bold')
    ax_osc.set_ylabel('Oscillation Strength', fontweight='bold')
    ax_osc.set_title('Temporal Variability', fontweight='bold', fontsize=9)
    ax_osc.grid(axis='y', alpha=0.3)
    ax_osc.set_xticklabels([f'F{i+1}' for i in range(N_CLUSTERS)])

    # Final density
    ax_final = fig.add_subplot(gs[2, 3])
    final_by_cluster = [features['final_density'][labels == i] for i in range(N_CLUSTERS)]
    bp4 = ax_final.boxplot(final_by_cluster, patch_artist=True, widths=0.6,
                           boxprops=dict(linewidth=1.0),
                           whiskerprops=dict(linewidth=1.0),
                           capprops=dict(linewidth=1.0),
                           medianprops=dict(linewidth=1.5, color='red'))

    for patch, color in zip(bp4['boxes'], COLORS_FAMILIES):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_final.set_xlabel('Family', fontweight='bold')
    ax_final.set_ylabel('Final Density', fontweight='bold')
    ax_final.set_title('Equilibrium Density', fontweight='bold', fontsize=9)
    ax_final.grid(axis='y', alpha=0.3)
    ax_final.set_xticklabels([f'F{i+1}' for i in range(N_CLUSTERS)])

    # Peak density
    ax_peak = fig.add_subplot(gs[2, 4])
    peak_by_cluster = [features['peak_density'][labels == i] for i in range(N_CLUSTERS)]
    bp5 = ax_peak.boxplot(peak_by_cluster, patch_artist=True, widths=0.6,
                          boxprops=dict(linewidth=1.0),
                          whiskerprops=dict(linewidth=1.0),
                          capprops=dict(linewidth=1.0),
                          medianprops=dict(linewidth=1.5, color='red'))

    for patch, color in zip(bp5['boxes'], COLORS_FAMILIES):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_peak.set_xlabel('Family', fontweight='bold')
    ax_peak.set_ylabel('Peak Density', fontweight='bold')
    ax_peak.set_title('Maximum Population', fontweight='bold', fontsize=9)
    ax_peak.grid(axis='y', alpha=0.3)
    ax_peak.set_xticklabels([f'F{i+1}' for i in range(N_CLUSTERS)])

    # Save figure
    output_png = f'{OUTPUT_DIR}fig_latent_space_families.png'
    output_pdf = f'{OUTPUT_DIR}fig_latent_space_families.pdf'

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"\n✓ Figure saved:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    plt.close()

    return features


def print_family_statistics(labels, features):
    """Print detailed statistics for each family."""
    print("\n" + "="*80)
    print("LATENT SPACE FAMILY ANALYSIS")
    print("="*80)

    for cluster_id in range(N_CLUSTERS):
        mask = labels == cluster_id
        n_samples = np.sum(mask)

        print(f"\nFamily {cluster_id+1} (n={n_samples}, {n_samples/len(labels)*100:.1f}%):")
        print(f"  Species Richness:     {np.mean(features['species_richness'][mask]):.2f} ± {np.std(features['species_richness'][mask]):.2f}")
        print(f"  Mean Density:         {np.mean(features['mean_density'][mask]):.3f} ± {np.std(features['mean_density'][mask]):.3f}")
        print(f"  Oscillation Strength: {np.mean(features['oscillation_strength'][mask]):.3f} ± {np.std(features['oscillation_strength'][mask]):.3f}")
        print(f"  Final Density:        {np.mean(features['final_density'][mask]):.3f} ± {np.std(features['final_density'][mask]):.3f}")
        print(f"  Peak Density:         {np.mean(features['peak_density'][mask]):.3f} ± {np.std(features['peak_density'][mask]):.3f}")

    print("\n" + "="*80)


def main():
    print("="*80)
    print("LATENT SPACE FAMILY ANALYSIS")
    print("Exploring whether different latent regions encode different dynamics")
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

    # Subsample for efficiency
    n_samples = min(N_SAMPLES_TO_ENCODE, len(test_data))
    indices = np.random.choice(len(test_data), size=n_samples, replace=False)
    test_data_subset = [test_data[i] for i in indices]
    test_max_vals_subset = [test_max_vals[i] for i in indices]

    print(f"Using {n_samples} samples for analysis")

    # Encode to latent space
    latent_codes, reconstructions = encode_test_set_to_latent(
        model, test_data_subset, test_max_vals_subset, n_samples
    )

    # Cluster latent space
    labels, kmeans = cluster_latent_space(latent_codes, N_CLUSTERS)

    # Generate figure
    features = generate_latent_family_figure(latent_codes, labels, reconstructions, kmeans)

    # Print statistics
    print_family_statistics(labels, features)

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
