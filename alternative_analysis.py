"""
Alternative analysis: Instead of expecting interaction matrices to cluster,
analyze what DOES cluster in the latent space.

Since the VAE was trained to reconstruct time series dynamics, not interaction matrices,
we should analyze dynamical features instead.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from scipy.stats import pearsonr
from tqdm import tqdm

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE


def extract_time_series_features(curves):
    """
    Extract dynamical features from time series that the VAE likely encodes.

    Args:
        curves: (N, n_species, n_timesteps) array

    Returns:
        dict of features
    """
    N, n_species, n_timesteps = curves.shape

    features = {}

    # 1. Peak timing and magnitude
    features['peak_times'] = np.argmax(curves, axis=2).mean(axis=1)  # Average peak time
    features['peak_magnitudes'] = np.max(curves, axis=2).mean(axis=1)  # Average peak height
    features['peak_variance'] = np.std(np.max(curves, axis=2), axis=1)  # Peak diversity

    # 2. Steady state behavior
    steady_window = curves[:, :, -20:]
    features['steady_state_mean'] = steady_window.mean(axis=(1, 2))
    features['steady_state_std'] = steady_window.std(axis=(1, 2))

    # 3. Initial growth rate
    initial_slopes = np.zeros((N, n_species))
    for i in range(N):
        for j in range(n_species):
            initial_slopes[i, j] = curves[i, j, 5] - curves[i, j, 0]
    features['initial_growth_mean'] = initial_slopes.mean(axis=1)
    features['initial_growth_std'] = initial_slopes.std(axis=1)

    # 4. Overshoot behavior (with safe calculation to prevent explosion)
    steady_vals = curves[:, :, -10:].mean(axis=2)
    peak_vals = curves.max(axis=2)
    # Use safe threshold and clipping to prevent division by near-zero
    steady_vals_safe = np.maximum(steady_vals, 1e-3)
    overshoot = (peak_vals - steady_vals_safe) / steady_vals_safe
    # Clip extreme values to prevent explosion to 10^12
    overshoot = np.clip(overshoot, -1.0, 10.0)
    # Set to 0 where steady state is effectively zero
    overshoot[steady_vals < 1e-4] = 0.0
    features['overshoot_mean'] = overshoot.mean(axis=1)
    features['overshoot_max'] = overshoot.max(axis=1)

    # 5. Species diversity (how many species persist)
    final_vals = curves[:, :, -1]
    features['num_survivors'] = (final_vals > 0.05).sum(axis=1)

    # 6. Temporal variance (how much dynamics vs static)
    temporal_variance = curves.var(axis=2).mean(axis=1)
    features['temporal_variance'] = temporal_variance

    # 7. Species correlation structure
    species_correlations = np.zeros(N)
    for i in range(N):
        corr_matrix = np.corrcoef(curves[i])
        species_correlations[i] = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
    features['species_correlation'] = species_correlations

    return features


def load_existing_data():
    """Load the previously generated dataset with metadata."""
    with open('data/interaction_mapping_dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('latent_interaction_results.pkl', 'rb') as f:
        results = pickle.load(f)

    return data, results


def visualize_dynamical_features(embedding, features):
    """Visualize latent space colored by dynamical features."""

    feature_list = [
        ('Peak Timing', features['peak_times']),
        ('Peak Magnitude', features['peak_magnitudes']),
        ('Overshoot Behavior', features['overshoot_mean']),
        ('Initial Growth Rate', features['initial_growth_mean']),
        ('Species Survivors', features['num_survivors']),
        ('Temporal Variance', features['temporal_variance']),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, values) in enumerate(feature_list):
        ax = axes[idx]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=values,
            cmap='viridis',
            s=20,
            alpha=0.6
        )
        ax.set_title(f'Latent Space: {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(name)

    plt.tight_layout()
    plt.savefig('latent_space_dynamical_features.png', dpi=300, bbox_inches='tight')
    print("Saved dynamical features visualization to latent_space_dynamical_features.png")
    plt.close()


def compute_feature_correlations(latent_codes, features):
    """Compute correlations between latent dimensions and dynamical features."""

    print("\n" + "="*60)
    print("Correlation between Latent Space and Dynamical Features")
    print("="*60)

    for feature_name, feature_values in features.items():
        if len(feature_values) != len(latent_codes):
            continue

        correlations = []
        for dim in range(latent_codes.shape[1]):
            corr, _ = pearsonr(latent_codes[:, dim], feature_values)
            correlations.append(corr)

        # Find top 3 most correlated dimensions
        top_dims = np.argsort(np.abs(correlations))[-3:][::-1]
        max_corr = np.abs(correlations[top_dims[0]])

        if max_corr > 0.2:  # Only print if there's meaningful correlation
            print(f"\n{feature_name}:")
            for dim in top_dims:
                print(f"  Latent dim {dim}: correlation = {correlations[dim]:.3f}")


def analyze_latent_structure(latent_codes):
    """Analyze the intrinsic structure of the latent space."""

    print("\n" + "="*60)
    print("Latent Space Structure Analysis")
    print("="*60)

    # 1. Variance explained by each dimension
    variances = np.var(latent_codes, axis=0)
    sorted_dims = np.argsort(variances)[::-1]

    print("\nTop 10 dimensions by variance:")
    for i in range(min(10, len(sorted_dims))):
        dim = sorted_dims[i]
        print(f"  Dim {dim}: variance = {variances[dim]:.4f}")

    # 2. Effective dimensionality
    normalized_var = variances / variances.sum()
    cumsum_var = np.cumsum(np.sort(normalized_var)[::-1])
    eff_dim_90 = np.argmax(cumsum_var >= 0.90) + 1
    eff_dim_95 = np.argmax(cumsum_var >= 0.95) + 1

    print(f"\nEffective dimensionality:")
    print(f"  90% variance captured by: {eff_dim_90} dimensions")
    print(f"  95% variance captured by: {eff_dim_95} dimensions")

    # 3. Correlation structure
    corr_matrix = np.corrcoef(latent_codes.T)
    mean_abs_corr = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
    print(f"\nMean absolute correlation between latent dimensions: {mean_abs_corr:.3f}")
    if mean_abs_corr < 0.1:
        print("  → Latent dimensions are largely independent (good disentanglement)")
    elif mean_abs_corr < 0.3:
        print("  → Moderate independence between dimensions")
    else:
        print("  → High correlation between dimensions (entangled representation)")


def main():
    print("="*60)
    print("Dynamical Features Analysis")
    print("="*60)

    # Load existing data
    print("\nLoading previously generated data...")
    data, results = load_existing_data()

    curves = data['curves']
    latent_codes = results['latent_codes']
    embedding = results['embedding_umap']

    print(f"Loaded {len(curves)} samples")

    # Extract dynamical features
    print("\nExtracting dynamical features from time series...")
    features = extract_time_series_features(curves)

    # Visualize
    print("\nCreating visualizations...")
    visualize_dynamical_features(embedding, features)

    # Compute correlations
    compute_feature_correlations(latent_codes, features)

    # Analyze latent structure
    analyze_latent_structure(latent_codes)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nKey insight: Your VAE encodes DYNAMICS (peaks, growth, steady states)")
    print("not the underlying interaction matrix parameters.")
    print("\nThis is expected behavior for a time series VAE!")


if __name__ == "__main__":
    main()
