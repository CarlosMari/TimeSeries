"""
Create a correlation heatmap between 50 latent dimensions and interaction matrix properties.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE

# --- Configuration ---
MODEL_PATH = 'model_ckpts/model_new_arch.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_DIR = Path("Figures Model 50")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data_and_encode():
    """Load dataset and encode to latent space."""
    print("Loading dataset...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    print(f"Dataset has {len(data['curves'])} samples")

    # Load trained model
    print(f"Loading model from {MODEL_PATH}...")
    model_config['latent_dim'] = 50  # Ensure we're using 50 dimensions
    model = LSTM_VAE(config=model_config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded - latent dim: {model.latent_dim}")

    # Encode to latent space
    print("Encoding curves to latent space...")
    curves_tensor = torch.tensor(data['curves'], dtype=torch.float32)
    latent_codes = []

    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(curves_tensor), batch_size):
            batch = curves_tensor[i:i+batch_size].to(DEVICE)
            # Get encoder output (mu only, not sampling)
            X_rnn = batch.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(X_rnn)
            encoded = h_n.permute(1, 0, 2).contiguous().view(batch.size(0), -1)
            mu = model.fc_mu(encoded)
            latent_codes.append(mu.cpu())

    latent_codes = torch.cat(latent_codes, dim=0).numpy()
    print(f"Latent codes shape: {latent_codes.shape}")

    return latent_codes, data['metadata']


def extract_interaction_properties(metadata_list):
    """Extract various properties from interaction matrices."""
    n_samples = len(metadata_list)
    n_species = metadata_list[0]['interaction_matrix'].shape[0]

    properties = {}

    # Global statistics
    properties['Max Eigenvalue'] = np.array([m['max_eigenvalue'] for m in metadata_list])
    properties['Mean Interaction'] = np.array([m['mean_interaction'] for m in metadata_list])
    properties['Diagonal Strength'] = np.array([m['diagonal_strength'] for m in metadata_list])
    properties['Stability Measure'] = np.array([m['stability_measure'] for m in metadata_list])
    properties['Interaction Std'] = np.array([m['interaction_std'] for m in metadata_list])

    # Individual growth rates
    for i in range(n_species):
        properties[f'Growth Rate {i}'] = np.array([m['growth_rates'][i] for m in metadata_list])

    # Diagonal elements (self-regulation)
    for i in range(n_species):
        properties[f'Self-regulation {i}'] = np.array([m['interaction_matrix'][i, i] for m in metadata_list])

    # Specific off-diagonal interactions (representative subset)
    # Include a few key interactions to avoid overcrowding
    key_interactions = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    for i, j in key_interactions:
        properties[f'Interaction [{i},{j}]'] = np.array([m['interaction_matrix'][i, j] for m in metadata_list])

    # Steady state values
    for i in range(n_species):
        properties[f'Steady State {i}'] = np.array([m['steady_state'][i] for m in metadata_list])

    return properties


def compute_correlation_matrix(latent_codes, properties_dict, method='pearson'):
    """Compute correlation matrix between latent dimensions and properties."""
    n_latent = latent_codes.shape[1]
    n_properties = len(properties_dict)

    correlation_matrix = np.zeros((n_properties, n_latent))
    p_values = np.zeros((n_properties, n_latent))

    property_names = list(properties_dict.keys())

    print(f"\nComputing {method} correlations...")
    for i, prop_name in enumerate(property_names):
        prop_values = properties_dict[prop_name]
        for j in range(n_latent):
            if method == 'pearson':
                corr, pval = pearsonr(latent_codes[:, j], prop_values)
            else:  # spearman
                corr, pval = spearmanr(latent_codes[:, j], prop_values)
            correlation_matrix[i, j] = corr
            p_values[i, j] = pval

    return correlation_matrix, p_values, property_names


def plot_correlation_heatmap(correlation_matrix, property_names, method='pearson',
                             significance_mask=None):
    """Create a comprehensive correlation heatmap."""

    n_properties, n_latent = correlation_matrix.shape

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1, interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(n_latent))
    ax.set_yticks(np.arange(n_properties))
    ax.set_xticklabels([f'L{i}' for i in range(n_latent)], fontsize=8)
    ax.set_yticklabels(property_names, fontsize=9)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{method.capitalize()} Correlation', fontsize=12)

    # Add significance markers (if provided)
    if significance_mask is not None:
        for i in range(n_properties):
            for j in range(n_latent):
                if significance_mask[i, j]:
                    ax.text(j, i, '*', ha="center", va="center",
                           color="black", fontsize=6, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Latent Dimensions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Interaction Matrix Properties', fontsize=14, fontweight='bold')
    ax.set_title(f'Correlation Heatmap: 50 Latent Dimensions vs GLV Properties\n({method.capitalize()} correlation, * indicates p<0.01)',
                fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.set_xticks(np.arange(n_latent) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_properties) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.3, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_top_correlations(correlation_matrix, property_names, top_k=10):
    """Plot bar chart of top absolute correlations."""

    # Flatten correlation matrix and find top correlations
    n_properties, n_latent = correlation_matrix.shape

    correlations_flat = []
    labels_flat = []

    for i in range(n_properties):
        for j in range(n_latent):
            correlations_flat.append(correlation_matrix[i, j])
            labels_flat.append(f'{property_names[i]} - L{j}')

    correlations_flat = np.array(correlations_flat)

    # Get top k by absolute value
    top_indices = np.argsort(np.abs(correlations_flat))[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(top_k)
    colors = ['red' if correlations_flat[i] < 0 else 'blue' for i in top_indices]

    ax.barh(y_pos, correlations_flat[top_indices], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels_flat[i] for i in top_indices], fontsize=10)
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Strongest Correlations', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    return fig


def plot_correlation_summary(correlation_matrix, property_names):
    """Create summary statistics of correlations."""

    # Compute statistics per property and per latent dimension
    max_abs_corr_per_property = np.max(np.abs(correlation_matrix), axis=1)
    max_abs_corr_per_latent = np.max(np.abs(correlation_matrix), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Max absolute correlation per property
    ax1 = axes[0]
    y_pos = np.arange(len(property_names))
    ax1.barh(y_pos, max_abs_corr_per_property, color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(property_names, fontsize=9)
    ax1.set_xlabel('Max |Correlation|', fontsize=12)
    ax1.set_title('Maximum Absolute Correlation per Property', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Max absolute correlation per latent dimension
    ax2 = axes[1]
    x_pos = np.arange(len(max_abs_corr_per_latent))
    ax2.bar(x_pos, max_abs_corr_per_latent, color='coral', alpha=0.7)
    ax2.set_xticks(x_pos[::5])
    ax2.set_xticklabels([f'L{i}' for i in range(0, len(max_abs_corr_per_latent), 5)], fontsize=9)
    ax2.set_ylabel('Max |Correlation|', fontsize=12)
    ax2.set_xlabel('Latent Dimension', fontsize=12)
    ax2.set_title('Maximum Absolute Correlation per Latent Dimension', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig


def main():
    print("="*60)
    print("Latent Dimension Correlation Analysis (50D)")
    print("="*60)

    # Load data and encode
    latent_codes, metadata_list = load_data_and_encode()

    # Extract properties
    print("\nExtracting interaction matrix properties...")
    properties_dict = extract_interaction_properties(metadata_list)
    print(f"Extracted {len(properties_dict)} properties")

    # Compute correlations (both Pearson and Spearman)
    for method in ['pearson', 'spearman']:
        print(f"\n{'='*60}")
        print(f"Computing {method.capitalize()} correlations...")
        print('='*60)

        corr_matrix, p_values, property_names = compute_correlation_matrix(
            latent_codes, properties_dict, method=method
        )

        # Create significance mask (p < 0.01)
        sig_mask = p_values < 0.01

        # Print summary statistics
        print(f"\nCorrelation matrix shape: {corr_matrix.shape}")
        print(f"Max absolute correlation: {np.max(np.abs(corr_matrix)):.4f}")
        print(f"Mean absolute correlation: {np.mean(np.abs(corr_matrix)):.4f}")
        print(f"Significant correlations (p<0.01): {np.sum(sig_mask)} / {corr_matrix.size}")
        print(f"Percentage significant: {100*np.sum(sig_mask)/corr_matrix.size:.1f}%")

        # Find strongest correlations
        print(f"\nTop 5 strongest correlations:")
        flat_idx = np.argsort(np.abs(corr_matrix.flatten()))[-5:][::-1]
        for idx in flat_idx:
            i, j = np.unravel_index(idx, corr_matrix.shape)
            print(f"  {property_names[i]} - L{j}: {corr_matrix[i, j]:.4f} (p={p_values[i, j]:.2e})")

        # Create visualizations
        print(f"\nGenerating {method} visualizations...")

        # 1. Main heatmap
        fig1 = plot_correlation_heatmap(corr_matrix, property_names, method, sig_mask)
        path1 = OUTPUT_DIR / f"Fig_Latent_Correlation_Heatmap_{method}.png"
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        print(f"Saved: {path1}")

        path1_pdf = OUTPUT_DIR / f"Fig_Latent_Correlation_Heatmap_{method}.pdf"
        fig1.savefig(path1_pdf, bbox_inches='tight')
        print(f"Saved: {path1_pdf}")
        plt.close(fig1)

        # 2. Top correlations bar chart
        fig2 = plot_top_correlations(corr_matrix, property_names, top_k=15)
        path2 = OUTPUT_DIR / f"Fig_Top_Correlations_{method}.png"
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        print(f"Saved: {path2}")
        plt.close(fig2)

        # 3. Summary statistics
        fig3 = plot_correlation_summary(corr_matrix, property_names)
        path3 = OUTPUT_DIR / f"Fig_Correlation_Summary_{method}.png"
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        print(f"Saved: {path3}")
        plt.close(fig3)

        # Save numerical results
        results = {
            'correlation_matrix': corr_matrix,
            'p_values': p_values,
            'property_names': property_names,
            'latent_codes': latent_codes,
            'method': method
        }
        results_path = OUTPUT_DIR / f"latent_correlation_results_{method}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results: {results_path}")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nGenerated files in 'Figures Model 50/':")
    print("  - Fig_Latent_Correlation_Heatmap_[pearson/spearman].png/pdf")
    print("  - Fig_Top_Correlations_[pearson/spearman].png")
    print("  - Fig_Correlation_Summary_[pearson/spearman].png")
    print("  - latent_correlation_results_[pearson/spearman].pkl")


if __name__ == "__main__":
    main()
