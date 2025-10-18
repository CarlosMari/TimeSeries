"""
Map GLV interaction matrices to VAE latent space to analyze if similar dynamics cluster together.

This script:
1. Generates GLV time series with known interaction matrices
2. Encodes them using trained CVAE model
3. Visualizes latent space colored by interaction matrix properties
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm
import umap
from sklearn.decomposition import PCA

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE

# --- Configuration ---
NUM_SAMPLES = 2000  # Number of GLV systems to generate
MODEL_PATH = 'model_ckpts/model_new_arch.pth'  # Update this path
SEED = 42


def gLV(t, x_t, r, A, K):
    """Generalized Lotka-Volterra dynamics."""
    return x_t * (r * (1 - x_t / K) + A @ x_t)


def generate_glv_with_metadata(myseed, species=7, tmax=20, n_points=129):
    """
    Generate GLV time series and return both the curve and interaction matrix.

    Returns:
        curves: (species, n_points) time series
        metadata: dict with interaction matrix and derived properties
    """
    np.random.seed(myseed)

    def lotka_volterra(t, x, params):
        K = len(x)
        r = params[:K]
        b = np.array(params[K:]).reshape((K, K)).T
        dxdt = np.zeros(K)
        for i in range(K):
            dxdt[i] = r[i] * x[i]
            for j in range(K):
                dxdt[i] += b[i, j] * x[i] * x[j]
        return dxdt

    # Generate parameters
    flag = False
    c = 0
    while not flag and c < 500:
        r0 = np.random.exponential(scale=2.0, size=species)
        a0 = np.random.randn(species, species)
        for i in range(species):
            a0[i, i] = -np.random.exponential(scale=2.0)

        params = np.concatenate([r0, a0.T.flatten()])
        xss = np.linalg.solve(a0, -r0)
        d0 = np.diag(xss)
        eigenvalues = np.real(np.linalg.eigvals(d0 @ a0))

        flag = np.all(eigenvalues <= 0) and np.all(xss > 0)
        c += 1

    if not flag:
        return None, None

    initial_condition = np.random.exponential(scale=0.1, size=species)

    # Solve ODE
    times = np.linspace(0, tmax, n_points)
    sol = integrate.solve_ivp(
        fun=lambda t, y: lotka_volterra(t, y, params),
        t_span=(0, tmax),
        y0=initial_condition,
        t_eval=times,
        method='RK45'
    )

    curves = sol.y

    # Add lognormal noise
    SIGMA = 0.01
    scaling_factor = float(np.exp(SIGMA**2 / 2))
    noise = np.random.lognormal(mean=0, sigma=SIGMA, size=curves.shape)
    centered_noise = noise / scaling_factor
    curves = curves * centered_noise

    # Check quality criteria
    steady_states = np.mean(curves[:, -10:], axis=1)
    overshoot_flags = np.max(curves, axis=1) > 1.2 * steady_states
    overshoot_count = np.sum(overshoot_flags)

    if np.isnan(curves).any() or curves[curves > 3.0].any() or \
       np.any(np.max(curves, axis=1) < 0.1) or overshoot_count < 3:
        return None, None

    # Calculate interaction matrix properties
    metadata = {
        'interaction_matrix': a0,
        'growth_rates': r0,
        'steady_state': xss,
        'eigenvalues': eigenvalues,
        'max_eigenvalue': np.max(eigenvalues),
        'mean_interaction': np.mean(np.abs(a0[~np.eye(species, dtype=bool)])),  # Off-diagonal
        'interaction_std': np.std(a0[~np.eye(species, dtype=bool)]),
        'diagonal_strength': np.mean(np.abs(np.diag(a0))),
        'stability_measure': -np.max(eigenvalues),  # More negative = more stable
    }

    return curves, metadata


def preprocess_curve(curve):
    """Preprocess a single curve to match training data format."""
    # Family normalization
    family_max = np.max(curve)
    if family_max == 0:
        family_max = 1e-8
    curve_norm = curve / family_max

    # Sort by peak values
    max_vals = np.max(curve_norm, axis=1)
    sorted_idx = np.argsort(-max_vals)
    curve_sorted = curve_norm[sorted_idx]

    # Individual curve normalization
    curve_maxes = np.max(curve_sorted, axis=1, keepdims=True)
    curve_maxes[curve_maxes == 0] = 1e-8
    final_curve = curve_sorted / curve_maxes

    reconstruction_max_values = np.squeeze(curve_maxes)

    return final_curve, reconstruction_max_values, family_max


def generate_dataset_with_metadata(num_samples, seed):
    """Generate dataset with interaction matrix metadata."""
    np.random.seed(seed)

    curves_list = []
    max_values_list = []
    metadata_list = []

    pbar = tqdm(range(num_samples * 3), desc='Generating GLV systems')
    successful = 0

    for i in pbar:
        curve, metadata = generate_glv_with_metadata(seed * 1000 + i)
        if curve is None:
            continue

        # Preprocess
        processed_curve, max_vals, family_max = preprocess_curve(curve)

        curves_list.append(processed_curve)
        max_values_list.append(max_vals)
        metadata_list.append(metadata)

        successful += 1
        pbar.set_postfix({'successful': successful})

        if successful >= num_samples:
            break

    return {
        'curves': np.array(curves_list),
        'max_values': np.array(max_values_list),
        'metadata': metadata_list
    }


def encode_dataset(model, curves_tensor):
    """Encode curves to latent space using the trained model."""
    model.eval()
    latent_codes = []

    with torch.no_grad():
        batch_size = 64
        for i in tqdm(range(0, len(curves_tensor), batch_size), desc='Encoding'):
            batch = curves_tensor[i:i+batch_size].to(DEVICE)
            # Get encoder output
            X_rnn = batch.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(X_rnn)
            encoded = h_n.permute(1, 0, 2).contiguous().view(batch.size(0), -1)
            mu = model.fc_mu(encoded)
            latent_codes.append(mu.cpu())

    return torch.cat(latent_codes, dim=0).numpy()


def visualize_latent_space(latent_codes, metadata_list, method='umap'):
    """Visualize latent space colored by interaction matrix properties."""

    # Extract properties for coloring
    properties = {
        'Max Eigenvalue': np.array([m['max_eigenvalue'] for m in metadata_list]),
        'Mean Interaction Strength': np.array([m['mean_interaction'] for m in metadata_list]),
        'Diagonal Strength': np.array([m['diagonal_strength'] for m in metadata_list]),
        'Stability': np.array([m['stability_measure'] for m in metadata_list]),
    }

    # Dimensionality reduction
    if method == 'umap':
        print("Computing UMAP projection...")
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(latent_codes)
    else:  # PCA
        print("Computing PCA projection...")
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(latent_codes)
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (prop_name, prop_values) in enumerate(properties.items()):
        ax = axes[idx]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=prop_values,
            cmap='viridis',
            s=20,
            alpha=0.6
        )
        ax.set_title(f'Latent Space colored by {prop_name}', fontsize=14)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(prop_name)

    plt.tight_layout()
    plt.savefig(f'latent_space_interaction_mapping_{method}.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to latent_space_interaction_mapping_{method}.png")
    plt.close()

    return embedding


def classify_interaction_type(metadata):
    """Classify interaction matrix into categories."""
    A = metadata['interaction_matrix']
    n = A.shape[0]

    # Get off-diagonal elements
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = A[off_diag_mask]

    mean_off_diag = np.mean(off_diag)
    positive_ratio = np.sum(off_diag > 0) / len(off_diag)

    # Classification logic
    if positive_ratio > 0.7:
        return 'Mutualism-dominated'
    elif positive_ratio < 0.3:
        return 'Competition-dominated'
    elif metadata['diagonal_strength'] > 3.0:
        return 'Strong self-regulation'
    elif metadata['diagonal_strength'] < 1.5:
        return 'Weak self-regulation'
    else:
        return 'Mixed interactions'


def visualize_interaction_clusters(embedding, metadata_list):
    """Visualize latent space with discrete interaction types."""

    # Classify each interaction matrix
    interaction_types = [classify_interaction_type(m) for m in metadata_list]
    unique_types = sorted(set(interaction_types))

    # Create color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    color_map = dict(zip(unique_types, colors))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each type
    for itype in unique_types:
        mask = np.array([t == itype for t in interaction_types])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color_map[itype]],
            label=itype,
            s=30,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Latent Space Colored by Interaction Type', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('latent_space_interaction_types.png', dpi=300, bbox_inches='tight')
    print("Saved interaction type clustering to latent_space_interaction_types.png")
    plt.close()

    # Print statistics
    print("\n" + "="*60)
    print("Interaction Type Distribution:")
    print("="*60)
    for itype in unique_types:
        count = interaction_types.count(itype)
        print(f"{itype}: {count} samples ({100*count/len(interaction_types):.1f}%)")

    return interaction_types


def analyze_specific_matrices(latent_codes, metadata_list, embedding):
    """Find and highlight specific interesting interaction matrices."""

    print("\n" + "="*60)
    print("Analyzing Specific Interaction Matrix Types")
    print("="*60)

    # Find examples of extreme cases
    indices = {
        'Most stable': np.argmax([m['stability_measure'] for m in metadata_list]),
        'Least stable': np.argmin([m['stability_measure'] for m in metadata_list]),
        'Strongest interactions': np.argmax([m['mean_interaction'] for m in metadata_list]),
        'Weakest interactions': np.argmin([m['mean_interaction'] for m in metadata_list]),
        'Strongest self-regulation': np.argmax([m['diagonal_strength'] for m in metadata_list]),
        'Weakest self-regulation': np.argmin([m['diagonal_strength'] for m in metadata_list]),
    }

    # Create figure showing these specific cases in latent space
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (label, sample_idx) in enumerate(indices.items()):
        ax = axes[idx]

        # Plot all points in gray
        ax.scatter(embedding[:, 0], embedding[:, 1], c='lightgray', s=10, alpha=0.3)

        # Highlight the specific point
        ax.scatter(
            embedding[sample_idx, 0],
            embedding[sample_idx, 1],
            c='red',
            s=200,
            marker='*',
            edgecolors='black',
            linewidth=2,
            zorder=5,
            label=label
        )

        # Add metadata text
        m = metadata_list[sample_idx]
        text = f"Stability: {m['stability_measure']:.3f}\n"
        text += f"Mean interact: {m['mean_interaction']:.3f}\n"
        text += f"Diag strength: {m['diagonal_strength']:.3f}"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(loc='upper right')

        # Print the interaction matrix
        print(f"\n{label} (sample {sample_idx}):")
        print("Interaction matrix:")
        print(metadata_list[sample_idx]['interaction_matrix'])
        print(f"Latent position: {embedding[sample_idx]}")

    plt.tight_layout()
    plt.savefig('latent_space_specific_matrices.png', dpi=300, bbox_inches='tight')
    print("\nSaved specific matrix locations to latent_space_specific_matrices.png")
    plt.close()

    return indices


def compute_distance_matrix_analysis(latent_codes, metadata_list):
    """Analyze if similar interaction matrices are close in latent space."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr

    print("\n" + "="*60)
    print("Distance Matrix Correlation Analysis")
    print("="*60)

    # Compute pairwise distances in latent space
    latent_distances = squareform(pdist(latent_codes, metric='euclidean'))

    # Compute pairwise distances between interaction matrices (Frobenius norm)
    interaction_matrices = np.array([m['interaction_matrix'].flatten() for m in metadata_list])
    interaction_distances = squareform(pdist(interaction_matrices, metric='euclidean'))

    # Compute correlation
    # Flatten upper triangular parts only (avoid duplicates)
    triu_indices = np.triu_indices_from(latent_distances, k=1)
    latent_dist_flat = latent_distances[triu_indices]
    interaction_dist_flat = interaction_distances[triu_indices]

    correlation, p_value = spearmanr(latent_dist_flat, interaction_dist_flat)

    print(f"\nSpearman correlation between latent distances and interaction matrix distances:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  P-value: {p_value:.4e}")

    if correlation > 0.3:
        print("  → Strong positive correlation! Similar interaction matrices cluster together.")
    elif correlation > 0.1:
        print("  → Moderate correlation. Some structure preserved.")
    else:
        print("  → Weak correlation. Latent space may prioritize dynamics over interaction structure.")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample points for visualization (all points would be too dense)
    sample_size = min(10000, len(latent_dist_flat))
    sample_idx = np.random.choice(len(latent_dist_flat), sample_size, replace=False)

    scatter = ax.scatter(
        interaction_dist_flat[sample_idx],
        latent_dist_flat[sample_idx],
        alpha=0.1,
        s=1,
        c='blue'
    )

    ax.set_xlabel('Interaction Matrix Distance (Frobenius norm)', fontsize=12)
    ax.set_ylabel('Latent Space Distance (Euclidean)', fontsize=12)
    ax.set_title(f'Correlation = {correlation:.3f} (p={p_value:.2e})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('latent_interaction_distance_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved distance correlation plot to latent_interaction_distance_correlation.png")
    plt.close()

    return correlation, p_value


def analyze_correlations(latent_codes, metadata_list):
    """Analyze correlations between latent dimensions and interaction properties."""
    properties = {
        'max_eigenvalue': np.array([m['max_eigenvalue'] for m in metadata_list]),
        'mean_interaction': np.array([m['mean_interaction'] for m in metadata_list]),
        'diagonal_strength': np.array([m['diagonal_strength'] for m in metadata_list]),
        'stability': np.array([m['stability_measure'] for m in metadata_list]),
    }

    print("\n" + "="*60)
    print("Correlation between latent dimensions and interaction properties")
    print("="*60)

    for prop_name, prop_values in properties.items():
        correlations = []
        for dim in range(latent_codes.shape[1]):
            corr = np.corrcoef(latent_codes[:, dim], prop_values)[0, 1]
            correlations.append(corr)

        # Find top 3 most correlated dimensions
        top_dims = np.argsort(np.abs(correlations))[-3:][::-1]

        print(f"\n{prop_name}:")
        for dim in top_dims:
            print(f"  Latent dim {dim}: correlation = {correlations[dim]:.3f}")


def main():
    print("="*60)
    print("Latent Space - Interaction Matrix Mapping Analysis")
    print("="*60)

    # 1. Generate dataset with metadata
    print(f"\nGenerating {NUM_SAMPLES} GLV systems with interaction matrices...")
    data = generate_dataset_with_metadata(NUM_SAMPLES, SEED)
    print(f"Successfully generated {len(data['curves'])} samples")

    # Save dataset for later use
    with open('data/interaction_mapping_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("Saved dataset to data/interaction_mapping_dataset.pkl")

    # 2. Load trained model
    print(f"\nLoading trained model from {MODEL_PATH}...")
    model = LSTM_VAE(config=model_config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully")

    # 3. Encode to latent space
    print("\nEncoding curves to latent space...")
    curves_tensor = torch.tensor(data['curves'], dtype=torch.float32)
    latent_codes = encode_dataset(model, curves_tensor)
    print(f"Latent codes shape: {latent_codes.shape}")

    # 4. Visualize
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    # UMAP visualization
    embedding_umap = visualize_latent_space(latent_codes, data['metadata'], method='umap')

    # PCA visualization
    embedding_pca = visualize_latent_space(latent_codes, data['metadata'], method='pca')

    # 5. Visualize interaction type clustering
    interaction_types = visualize_interaction_clusters(embedding_umap, data['metadata'])

    # 6. Analyze specific extreme matrices
    specific_indices = analyze_specific_matrices(latent_codes, data['metadata'], embedding_umap)

    # 7. Distance correlation analysis
    correlation, p_value = compute_distance_matrix_analysis(latent_codes, data['metadata'])

    # 8. Correlation analysis
    analyze_correlations(latent_codes, data['metadata'])

    # Save results
    results = {
        'latent_codes': latent_codes,
        'embedding_umap': embedding_umap,
        'embedding_pca': embedding_pca,
        'metadata': data['metadata'],
        'interaction_types': interaction_types,
        'distance_correlation': correlation,
        'specific_indices': specific_indices
    }
    with open('latent_interaction_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nSaved analysis results to latent_interaction_results.pkl")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nGenerated outputs:")
    print("  1. latent_space_interaction_mapping_umap.png - Continuous properties")
    print("  2. latent_space_interaction_mapping_pca.png - PCA projection")
    print("  3. latent_space_interaction_types.png - Discrete interaction types")
    print("  4. latent_space_specific_matrices.png - Extreme cases highlighted")
    print("  5. latent_interaction_distance_correlation.png - Distance preservation")
    print("  6. latent_interaction_results.pkl - All numerical results")


if __name__ == "__main__":
    main()
