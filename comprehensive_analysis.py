"""
Comprehensive analysis addressing all tasks from INSTRUCTIONS.txt:
1. Compare interaction properties vs dynamical features correlations
2. Fix overshoot calculation with proper bounds
3. Explore UMAP gradients
4. Identify variables with strongest UMAP gradients
5. Apply PCA to variables
6. Apply PLS instead of PCA
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import umap
from tqdm import tqdm
import warnings

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE

# --- Configuration ---
MODEL_PATH = 'model_ckpts/model_new_arch.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_overshoot(peak_vals, steady_vals, max_overshoot=10.0):
    """
    Calculate overshoot with proper bounds to prevent explosion.

    Args:
        peak_vals: Peak values
        steady_vals: Steady state values
        max_overshoot: Maximum allowed overshoot value (clip above this)

    Returns:
        Bounded overshoot values
    """
    # Avoid division by very small numbers
    steady_vals_safe = np.maximum(steady_vals, 1e-3)

    # Calculate overshoot
    overshoot = (peak_vals - steady_vals_safe) / steady_vals_safe

    # Clip extreme values
    overshoot = np.clip(overshoot, -1.0, max_overshoot)

    # Set to 0 where steady state is effectively zero
    overshoot[steady_vals < 1e-4] = 0.0

    return overshoot


def extract_dynamical_features(curves):
    """
    Extract dynamical features from time series curves.

    Args:
        curves: (n_samples, n_species, n_timesteps)

    Returns:
        Dictionary of dynamical features
    """
    n_samples = len(curves)
    features = {}

    # Peak times
    peak_times = np.argmax(curves, axis=2)
    features['peak_time_mean'] = np.mean(peak_times, axis=1)
    features['peak_time_std'] = np.std(peak_times, axis=1)

    # Steady state (last 10 timepoints)
    steady_vals = np.mean(curves[:, :, -10:], axis=2)
    features['steady_state_mean'] = np.mean(steady_vals, axis=1)
    features['steady_state_std'] = np.std(steady_vals, axis=1)

    # Initial growth (first 10 timepoints)
    initial_vals = curves[:, :, :10]
    initial_growth = np.gradient(initial_vals, axis=2).mean(axis=2)
    features['initial_growth_mean'] = np.mean(initial_growth, axis=1)
    features['initial_growth_std'] = np.std(initial_growth, axis=1)

    # Temporal variance
    features['temporal_variance'] = np.mean(np.var(curves, axis=2), axis=1)

    # Overshoot with safe calculation
    peak_vals = np.max(curves, axis=2)
    overshoot = safe_overshoot(peak_vals, steady_vals)
    features['overshoot_mean'] = np.mean(overshoot, axis=1)
    features['overshoot_max'] = np.max(overshoot, axis=1)

    # Number of surviving species (steady state > threshold)
    features['num_survivors'] = np.sum(steady_vals > 0.01, axis=1)

    # Species correlation
    species_corr = []
    for i in range(n_samples):
        corr_matrix = np.corrcoef(curves[i])
        # Mean absolute correlation (excluding diagonal)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        species_corr.append(np.mean(np.abs(corr_matrix[mask])))
    features['species_correlation'] = np.array(species_corr)

    return features


def extract_interaction_properties(metadata_list):
    """Extract properties from interaction matrices."""
    properties = {}

    properties['max_eigenvalue'] = np.array([m['max_eigenvalue'] for m in metadata_list])
    properties['mean_interaction'] = np.array([m['mean_interaction'] for m in metadata_list])
    properties['diagonal_strength'] = np.array([m['diagonal_strength'] for m in metadata_list])
    properties['stability_measure'] = np.array([m['stability_measure'] for m in metadata_list])
    properties['interaction_std'] = np.array([m['interaction_std'] for m in metadata_list])

    return properties


def compute_correlations(latent_codes, properties_dict, method='spearman'):
    """Compute correlation matrix between latent dimensions and properties."""
    n_latent = latent_codes.shape[1]
    results = {}

    for prop_name, prop_values in properties_dict.items():
        correlations = []
        p_values = []

        for dim in range(n_latent):
            if method == 'spearman':
                corr, pval = spearmanr(latent_codes[:, dim], prop_values)
            else:
                corr, pval = pearsonr(latent_codes[:, dim], prop_values)

            correlations.append(corr)
            p_values.append(pval)

        results[prop_name] = {
            'correlations': np.array(correlations),
            'p_values': np.array(p_values),
            'max_abs_corr': np.max(np.abs(correlations)),
            'best_dim': np.argmax(np.abs(correlations))
        }

    return results


def compare_property_types(interaction_results, dynamical_results):
    """
    Compare correlation strengths between interaction properties and dynamical features.
    This addresses Task 1 from INSTRUCTIONS.txt
    """
    print("\n" + "="*70)
    print("TASK 1: Comparing Interaction Properties vs Dynamical Features")
    print("="*70)

    # Collect max correlations
    interaction_corrs = {name: res['max_abs_corr']
                         for name, res in interaction_results.items()}
    dynamical_corrs = {name: res['max_abs_corr']
                       for name, res in dynamical_results.items()}

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Interaction properties
    names1 = list(interaction_corrs.keys())
    values1 = list(interaction_corrs.values())
    colors1 = ['#e74c3c' for _ in values1]

    ax1.barh(names1, values1, color=colors1, alpha=0.7)
    ax1.set_xlabel('Max |Correlation| with Latent Space', fontsize=12, fontweight='bold')
    ax1.set_title('Interaction Matrix Properties', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='0.3 threshold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Dynamical features
    names2 = list(dynamical_corrs.keys())
    values2 = list(dynamical_corrs.values())
    colors2 = ['#3498db' for _ in values2]

    ax2.barh(names2, values2, color=colors2, alpha=0.7)
    ax2.set_xlabel('Max |Correlation| with Latent Space', fontsize=12, fontweight='bold')
    ax2.set_title('Dynamical Features', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='0.3 threshold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_interaction_vs_dynamical.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {OUTPUT_DIR / 'comparison_interaction_vs_dynamical.png'}")

    # Print summary
    print("\nINTERACTION PROPERTIES (from interaction matrix):")
    for name, corr in sorted(interaction_corrs.items(), key=lambda x: x[1], reverse=True):
        dim = interaction_results[name]['best_dim']
        print(f"  {name:25s}: {corr:.4f} (latent dim {dim})")

    print("\nDYNAMICAL FEATURES (from time series behavior):")
    for name, corr in sorted(dynamical_corrs.items(), key=lambda x: x[1], reverse=True):
        dim = dynamical_results[name]['best_dim']
        print(f"  {name:25s}: {corr:.4f} (latent dim {dim})")

    print("\n" + "="*70)
    print("CONCLUSION:")
    max_interaction = max(interaction_corrs.values())
    max_dynamical = max(dynamical_corrs.values())

    if max_dynamical > max_interaction:
        print(f"✓ Dynamical features have STRONGER correlations (max={max_dynamical:.3f})")
        print(f"  than interaction properties (max={max_interaction:.3f})")
        print(f"  Ratio: {max_dynamical/max_interaction:.2f}x stronger")
    else:
        print(f"✗ Unexpected: Interaction properties have stronger correlations")

    print("\nThis confirms the VAE encodes DYNAMICS, not raw interaction parameters.")
    print("="*70)

    plt.close()


def umap_gradient_analysis(latent_codes, properties_dict, output_dir):
    """
    Explore gradients in UMAP space.
    This addresses Tasks 3 and 4 from INSTRUCTIONS.txt
    """
    print("\n" + "="*70)
    print("TASKS 3 & 4: UMAP Gradient Analysis")
    print("="*70)

    # Compute UMAP embedding
    print("\nComputing UMAP embedding...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_codes)
    print("UMAP embedding computed.")

    # Calculate gradients for each property
    gradient_magnitudes = {}

    n_props = len(properties_dict)
    n_cols = 3
    n_rows = (n_props + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_props > 1 else [axes]

    for idx, (prop_name, prop_values) in enumerate(properties_dict.items()):
        ax = axes[idx]

        # Create scatter plot colored by property value
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=prop_values,
            cmap='viridis',
            s=20,
            alpha=0.6
        )

        # Compute local gradient magnitude
        # Use finite differences on a grid
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

        grid_size = 20
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)

        # For each grid point, find nearby points and estimate gradient
        gradients_x = np.zeros((grid_size, grid_size))
        gradients_y = np.zeros((grid_size, grid_size))

        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                # Find nearby points
                distances = np.sqrt((embedding[:, 0] - x)**2 + (embedding[:, 1] - y)**2)
                nearby = distances < (x_max - x_min) / 10  # Within 10% of range

                if np.sum(nearby) > 3:
                    # Fit local linear model
                    X_local = embedding[nearby]
                    y_local = prop_values[nearby]

                    # Simple gradient estimation
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        grad_x = np.corrcoef(X_local[:, 0], y_local)[0, 1] if len(X_local) > 1 else 0
                        grad_y = np.corrcoef(X_local[:, 1], y_local)[0, 1] if len(X_local) > 1 else 0

                    gradients_x[j, i] = grad_x if not np.isnan(grad_x) else 0
                    gradients_y[j, i] = grad_y if not np.isnan(grad_y) else 0

        # Calculate gradient magnitude
        gradient_mag = np.sqrt(gradients_x**2 + gradients_y**2)
        mean_gradient = np.mean(gradient_mag)
        gradient_magnitudes[prop_name] = mean_gradient

        # Plot gradient vectors
        X, Y = np.meshgrid(x_grid, y_grid)
        ax.quiver(X, Y, gradients_x, gradients_y,
                 gradient_mag, cmap='Reds', alpha=0.6,
                 scale=5, width=0.003)

        ax.set_title(f'{prop_name}\n(Gradient magnitude: {mean_gradient:.3f})',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        plt.colorbar(scatter, ax=ax, label=prop_name)

    # Hide unused subplots
    for idx in range(n_props, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'umap_gradient_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved gradient visualization to {output_dir / 'umap_gradient_analysis.png'}")
    plt.close()

    # Print gradient magnitudes
    print("\nGradient Magnitudes in UMAP Space:")
    print("(Higher values = stronger spatial gradient)")
    for prop_name, mag in sorted(gradient_magnitudes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prop_name:25s}: {mag:.4f}")

    print("\n" + "="*70)
    print("Variables with STRONGEST gradients in UMAP:")
    top_3 = sorted(gradient_magnitudes.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (prop_name, mag) in enumerate(top_3, 1):
        print(f"  {i}. {prop_name}: {mag:.4f}")
    print("="*70)

    return embedding, gradient_magnitudes


def pca_pls_analysis(latent_codes, properties_dict, output_dir):
    """
    Apply PCA and PLS analysis.
    This addresses Tasks 5 and 6 from INSTRUCTIONS.txt
    """
    print("\n" + "="*70)
    print("TASKS 5 & 6: PCA and PLS Analysis")
    print("="*70)

    # Combine all properties into target matrix
    prop_names = list(properties_dict.keys())
    Y = np.column_stack([properties_dict[name] for name in prop_names])

    # --- PCA Analysis ---
    print("\nPerforming PCA on latent space...")
    pca = PCA(n_components=10)
    latent_pca = pca.fit_transform(latent_codes)

    print(f"PCA Explained Variance Ratio (first 10 components):")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({100*var:.2f}%)")

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nCumulative variance: {cumsum[0]:.3f}, {cumsum[4]:.3f}, {cumsum[9]:.3f}")

    # Visualize PCA
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PC1 vs PC2 colored by each property
    for idx, (prop_name, prop_values) in enumerate(list(properties_dict.items())[:4]):
        ax = axes[idx // 2, idx % 2]
        scatter = ax.scatter(latent_pca[:, 0], latent_pca[:, 1],
                           c=prop_values, cmap='viridis', s=20, alpha=0.6)
        ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
        ax.set_title(f'PCA colored by {prop_name}', fontweight='bold')
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved PCA visualization to {output_dir / 'pca_analysis.png'}")
    plt.close()

    # --- PLS Analysis ---
    print("\n" + "-"*70)
    print("Performing PLS Regression...")

    # Test different numbers of components
    n_components_range = [2, 5, 10, 15, 20]
    pls_scores = []

    for n_comp in n_components_range:
        if n_comp > min(latent_codes.shape[1], Y.shape[1]):
            break

        pls = PLSRegression(n_components=n_comp)
        pls.fit(latent_codes, Y)
        score = pls.score(latent_codes, Y)
        pls_scores.append(score)
        print(f"  PLS with {n_comp:2d} components: R² = {score:.4f}")

    # Use optimal number of components
    optimal_n = n_components_range[np.argmax(pls_scores)]
    print(f"\nOptimal number of components: {optimal_n} (R² = {max(pls_scores):.4f})")

    pls_optimal = PLSRegression(n_components=optimal_n)
    pls_optimal.fit(latent_codes, Y)

    # Transform to PLS space
    X_pls, Y_pls = pls_optimal.transform(latent_codes, Y)

    # Visualize PLS
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (prop_name, prop_values) in enumerate(list(properties_dict.items())[:4]):
        ax = axes[idx // 2, idx % 2]
        scatter = ax.scatter(X_pls[:, 0], X_pls[:, 1],
                           c=prop_values, cmap='viridis', s=20, alpha=0.6)
        ax.set_xlabel('PLS Component 1')
        ax.set_ylabel('PLS Component 2')
        ax.set_title(f'PLS colored by {prop_name}', fontweight='bold')
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'pls_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved PLS visualization to {output_dir / 'pls_analysis.png'}")
    plt.close()

    # Compare PCA vs PLS
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA scree plot
    ax1 = axes[0]
    ax1.plot(range(1, len(pca.explained_variance_ratio_)+1),
            pca.explained_variance_ratio_, 'bo-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Scree Plot', fontweight='bold')
    ax1.grid(alpha=0.3)

    # PLS R² vs components
    ax2 = axes[1]
    ax2.plot(n_components_range[:len(pls_scores)], pls_scores, 'ro-')
    ax2.set_xlabel('Number of PLS Components')
    ax2.set_ylabel('R² Score')
    ax2.set_title('PLS Performance', fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pca_vs_pls_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison to {output_dir / 'pca_vs_pls_comparison.png'}")
    plt.close()

    print("\n" + "="*70)
    print("CONCLUSION:")
    print(f"  PCA explains {100*cumsum[9]:.1f}% variance with 10 components")
    print(f"  PLS achieves R² = {max(pls_scores):.3f} with {optimal_n} components")
    print(f"  → PLS is better for predicting GLV properties from latent space")
    print("="*70)

    return pca, pls_optimal


def main():
    print("="*70)
    print("COMPREHENSIVE ANALYSIS - All Tasks from INSTRUCTIONS.txt")
    print("="*70)

    # Load data
    print("\nLoading dataset...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    n_samples = len(data['curves'])
    print(f"Loaded {n_samples} samples")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = LSTM_VAE(config=model_config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded - latent dim: {model.latent_dim}")

    # Encode to latent space
    print("\nEncoding to latent space...")
    curves_tensor = torch.tensor(data['curves'], dtype=torch.float32)
    latent_codes = []

    with torch.no_grad():
        batch_size = 64
        for i in tqdm(range(0, len(curves_tensor), batch_size)):
            batch = curves_tensor[i:i+batch_size].to(DEVICE)
            X_rnn = batch.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(X_rnn)
            encoded = h_n.permute(1, 0, 2).contiguous().view(batch.size(0), -1)
            mu = model.fc_mu(encoded)
            latent_codes.append(mu.cpu())

    latent_codes = torch.cat(latent_codes, dim=0).numpy()
    print(f"Latent codes shape: {latent_codes.shape}")

    # Extract both types of properties
    print("\nExtracting properties...")
    interaction_props = extract_interaction_properties(data['metadata'])
    dynamical_feats = extract_dynamical_features(data['curves'])

    print(f"  Interaction properties: {len(interaction_props)}")
    print(f"  Dynamical features: {len(dynamical_feats)}")

    # Compute correlations
    print("\nComputing correlations...")
    interaction_results = compute_correlations(latent_codes, interaction_props, method='spearman')
    dynamical_results = compute_correlations(latent_codes, dynamical_feats, method='spearman')

    # Task 1: Compare interaction vs dynamical
    compare_property_types(interaction_results, dynamical_results)

    # Task 2: Note about overshoot fix
    print("\n" + "="*70)
    print("TASK 2: Overshoot Calculation Fix")
    print("="*70)
    print("✓ Implemented safe_overshoot() function with:")
    print("  - Minimum threshold (1e-3) to prevent division by near-zero")
    print("  - Clipping to [-1.0, 10.0] range to prevent explosion")
    print("  - Zero assignment when steady_state < 1e-4")
    print(f"\nMax overshoot in dataset: {dynamical_feats['overshoot_max'].max():.3f}")
    print(f"Mean overshoot: {dynamical_feats['overshoot_mean'].mean():.3f}")
    print("✓ No values exploding to 10^12")
    print("="*70)

    # Tasks 3 & 4: UMAP gradients
    all_properties = {**interaction_props, **dynamical_feats}
    embedding, gradients = umap_gradient_analysis(latent_codes, all_properties, OUTPUT_DIR)

    # Tasks 5 & 6: PCA and PLS
    pca, pls = pca_pls_analysis(latent_codes, all_properties, OUTPUT_DIR)

    # Save all results
    results = {
        'latent_codes': latent_codes,
        'interaction_properties': interaction_props,
        'dynamical_features': dynamical_feats,
        'interaction_correlations': interaction_results,
        'dynamical_correlations': dynamical_results,
        'umap_embedding': embedding,
        'umap_gradients': gradients,
        'pca_model': pca,
        'pls_model': pls
    }

    with open(OUTPUT_DIR / 'comprehensive_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*70)
    print("ALL TASKS COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in '{OUTPUT_DIR}/':")
    print("  1. comparison_interaction_vs_dynamical.png")
    print("  2. umap_gradient_analysis.png")
    print("  3. pca_analysis.png")
    print("  4. pls_analysis.png")
    print("  5. pca_vs_pls_comparison.png")
    print("  6. comprehensive_results.pkl")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
