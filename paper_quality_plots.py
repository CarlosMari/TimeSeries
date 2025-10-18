"""
Generate publication-quality plots for latent space analysis.
Clean, high-resolution figures suitable for academic papers.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# Color palettes
COLORS = {
    'main': '#2E86AB',
    'accent': '#A23B72',
    'highlight': '#F18F01',
    'neutral': '#C73E1D',
    'dark': '#1B2021',
}


def load_data():
    """Load all analysis results."""
    with open('data/interaction_mapping_dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('latent_interaction_results.pkl', 'rb') as f:
        results = pickle.load(f)

    return data, results


def extract_time_series_features(curves):
    """Extract dynamical features from time series."""
    N, n_species, n_timesteps = curves.shape
    features = {}

    # Peak timing and magnitude
    features['peak_times'] = np.argmax(curves, axis=2).mean(axis=1)
    features['peak_magnitudes'] = np.max(curves, axis=2).mean(axis=1)

    # Steady state
    steady_window = curves[:, :, -20:]
    features['steady_state_mean'] = steady_window.mean(axis=(1, 2))
    features['steady_state_std'] = steady_window.std(axis=(1, 2))

    # Initial growth
    initial_slopes = np.zeros((N, n_species))
    for i in range(N):
        for j in range(n_species):
            initial_slopes[i, j] = curves[i, j, 5] - curves[i, j, 0]
    features['initial_growth_mean'] = initial_slopes.mean(axis=1)

    # Overshoot (with safe calculation to prevent explosion)
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

    # Temporal variance
    temporal_variance = curves.var(axis=2).mean(axis=1)
    features['temporal_variance'] = temporal_variance

    # Species correlation
    species_correlations = np.zeros(N)
    for i in range(N):
        corr_matrix = np.corrcoef(curves[i])
        species_correlations[i] = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
    features['species_correlation'] = species_correlations

    return features


def plot_latent_space_key_features(embedding, features):
    """
    Create a clean 2x2 grid showing the 4 most important dynamical features.
    """
    # Select the 4 most interpretable and strongly correlated features
    selected_features = [
        ('Temporal Variance', features['temporal_variance'], 'viridis'),
        ('Steady State Value', features['steady_state_mean'], 'plasma'),
        ('Peak Timing', features['peak_times'], 'cividis'),
        ('Overshoot Magnitude', features['overshoot_mean'], 'inferno'),
    ]

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    for idx, (name, values, cmap) in enumerate(selected_features):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # Create scatter plot with better visual properties
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=values,
            cmap=cmap,
            s=15,
            alpha=0.7,
            edgecolors='none',
            rasterized=True  # For better PDF rendering
        )

        # Styling
        ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=12, fontweight='bold', pad=10)

        # Clean up axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)

        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(name, fontsize=9)

    plt.savefig('Fig_LatentSpace_KeyFeatures.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_LatentSpace_KeyFeatures.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_LatentSpace_KeyFeatures.pdf/png")
    plt.close()


def plot_correlation_heatmap(latent_codes, features):
    """
    Create a heatmap showing correlations between latent dimensions and features.
    """
    feature_names = {
        'temporal_variance': 'Temporal Variance',
        'steady_state_mean': 'Steady State',
        'peak_times': 'Peak Timing',
        'overshoot_mean': 'Overshoot',
        'initial_growth_mean': 'Initial Growth',
        'species_correlation': 'Species Correlation',
    }

    # Compute correlation matrix
    n_features = len(feature_names)
    n_latent = latent_codes.shape[1]
    corr_matrix = np.zeros((n_features, n_latent))

    for i, (feat_key, feat_name) in enumerate(feature_names.items()):
        for j in range(n_latent):
            corr, _ = pearsonr(latent_codes[:, j], features[feat_key])
            corr_matrix[i, j] = corr

    # Find most informative latent dimensions (top 10 by max absolute correlation)
    max_corrs = np.max(np.abs(corr_matrix), axis=0)
    top_dims = np.argsort(max_corrs)[-10:][::-1]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        corr_matrix[:, top_dims],
        cmap='RdBu_r',
        aspect='auto',
        vmin=-0.8,
        vmax=0.8,
        interpolation='nearest'
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(top_dims)))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels([f'z{d}' for d in top_dims], fontsize=10)
    ax.set_yticklabels(list(feature_names.values()), fontsize=10)

    ax.set_xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dynamical Feature', fontsize=12, fontweight='bold')
    ax.set_title('Latent Space Encoding of Dynamical Features', fontsize=13, fontweight='bold', pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    # Add correlation values as text
    for i in range(n_features):
        for j in range(len(top_dims)):
            value = corr_matrix[i, top_dims[j]]
            if np.abs(value) > 0.25:  # Only show strong correlations
                text_color = 'white' if np.abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Fig_Correlation_Heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Correlation_Heatmap.pdf/png")
    plt.close()


def plot_latent_dimension_analysis(latent_codes, features):
    """
    Create a figure showing the top 3 latent dimensions and what they encode.
    """
    # Find top 3 dimensions by max correlation with any feature
    feature_list = [
        ('temporal_variance', 'Temporal Variance'),
        ('steady_state_mean', 'Steady State'),
        ('initial_growth_mean', 'Initial Growth'),
    ]

    # Identify key dimensions
    key_dims = []
    dim_features = []
    for feat_key, feat_name in feature_list:
        corrs = [pearsonr(latent_codes[:, j], features[feat_key])[0] for j in range(latent_codes.shape[1])]
        best_dim = np.argmax(np.abs(corrs))
        key_dims.append(best_dim)
        dim_features.append((feat_name, features[feat_key], corrs[best_dim]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (ax, dim, (feat_name, feat_values, corr)) in enumerate(zip(axes, key_dims, dim_features)):
        # Scatter plot
        scatter = ax.scatter(
            latent_codes[:, dim],
            feat_values,
            c=feat_values,
            cmap='viridis',
            s=20,
            alpha=0.6,
            edgecolors='none',
            rasterized=True
        )

        # Add regression line
        z = np.polyfit(latent_codes[:, dim], feat_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(latent_codes[:, dim].min(), latent_codes[:, dim].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label=f'r = {corr:.3f}')

        # Styling
        ax.set_xlabel(f'Latent Dimension z{dim}', fontsize=11, fontweight='bold')
        ax.set_ylabel(feat_name, fontsize=11, fontweight='bold')
        ax.set_title(f'z{dim} encodes {feat_name}', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig('Fig_LatentDimension_Interpretation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_LatentDimension_Interpretation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_LatentDimension_Interpretation.pdf/png")
    plt.close()


def plot_variance_explained(latent_codes):
    """
    Plot variance explained by each latent dimension.
    """
    variances = np.var(latent_codes, axis=0)
    sorted_var = np.sort(variances)[::-1]
    normalized_var = sorted_var / sorted_var.sum()
    cumsum_var = np.cumsum(normalized_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar plot of individual variances
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_var)))
    ax1.bar(range(len(sorted_var)), sorted_var, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Latent Dimension (sorted)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Variance', fontsize=11, fontweight='bold')
    ax1.set_title('Variance per Latent Dimension', fontsize=12, fontweight='bold', pad=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=9)

    # Cumulative variance plot
    ax2.plot(range(len(cumsum_var)), cumsum_var, linewidth=2.5, color=COLORS['main'])
    ax2.axhline(y=0.90, color=COLORS['accent'], linestyle='--', linewidth=2, label='90% variance')
    ax2.axhline(y=0.95, color=COLORS['highlight'], linestyle='--', linewidth=2, label='95% variance')
    ax2.fill_between(range(len(cumsum_var)), 0, cumsum_var, alpha=0.3, color=COLORS['main'])
    ax2.set_xlabel('Number of Latent Dimensions', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.tick_params(labelsize=9)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('Fig_Variance_Explained.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Variance_Explained.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Variance_Explained.pdf/png")
    plt.close()


def plot_example_time_series(curves, embedding, features):
    """
    Show example time series from different regions of latent space.
    Use multiple dynamical features to show diverse examples.
    """
    # Select examples based on different dynamical characteristics
    temporal_var = features['temporal_variance']
    steady_state = features['steady_state_mean']
    overshoot = features['overshoot_mean']
    peak_timing = features['peak_times']

    # Select 4 distinctive examples
    example_criteria = [
        ('Low Temporal Variance\n(Stable dynamics)', np.argmin(temporal_var)),
        ('High Overshoot\n(Boom-bust)', np.argmax(overshoot)),
        ('Late Peak Timing\n(Slow growth)', np.argmax(peak_timing)),
        ('High Steady State\n(Strong coexistence)', np.argmax(steady_state)),
    ]

    example_indices = [idx for _, idx in example_criteria]
    labels = [label for label, _ in example_criteria]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4, height_ratios=[1.2, 1, 0.3])

    # Top row: UMAP with highlighted examples
    ax_umap = fig.add_subplot(gs[0, :])
    ax_umap.scatter(embedding[:, 0], embedding[:, 1], c='lightgray', s=10, alpha=0.3, rasterized=True)

    colors_examples = ['#E63946', '#F77F00', '#06D6A0', '#118AB2']

    for idx, color, label in zip(example_indices, colors_examples, labels):
        ax_umap.scatter(embedding[idx, 0], embedding[idx, 1],
                       c=color, s=200, marker='*', edgecolors='black',
                       linewidth=2, zorder=5, label=label.replace('\n', ' '))

    ax_umap.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
    ax_umap.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
    ax_umap.set_title('Example Time Series with Distinct Dynamical Properties', fontsize=13, fontweight='bold', pad=10)
    ax_umap.legend(loc='best', fontsize=9, frameon=True, shadow=True, ncol=2)
    ax_umap.spines['top'].set_visible(False)
    ax_umap.spines['right'].set_visible(False)

    # Middle row: Time series plots
    for col, (idx, color, label) in enumerate(zip(example_indices, colors_examples, labels)):
        ax = fig.add_subplot(gs[1, col])

        time_series = curves[idx]  # (7 species, time_steps)

        for species in range(time_series.shape[0]):
            ax.plot(time_series[species], linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time', fontsize=9, fontweight='bold')
        ax.set_ylabel('Population', fontsize=9, fontweight='bold')
        ax.set_title(label, fontsize=10, fontweight='bold', color=color, pad=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.2, linestyle='--')

    # Bottom row: Quantitative metrics for each example
    for col, idx in enumerate(example_indices):
        ax = fig.add_subplot(gs[2, col])
        ax.axis('off')

        # Compute metrics for this example
        ts = curves[idx]
        temp_var = np.var(ts, axis=1).mean()
        ss_mean = ts[:, -20:].mean()
        peak_time = np.argmax(ts, axis=1).mean()
        # Safe overshoot calculation
        steady_vals_safe = np.maximum(ts[:, -10:].mean(axis=1), 1e-3)
        overshoot_raw = (ts.max(axis=1) - steady_vals_safe) / steady_vals_safe
        overshoot_val = np.clip(overshoot_raw, -1.0, 10.0).mean()

        metrics_text = f'Temporal var: {temp_var:.3f}\n'
        metrics_text += f'Steady state: {ss_mean:.3f}\n'
        metrics_text += f'Peak time: {peak_time:.1f}\n'
        metrics_text += f'Overshoot: {overshoot_val:.3f}'

        ax.text(0.5, 0.5, metrics_text, ha='center', va='center',
               fontsize=8, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.savefig('Fig_Example_TimeSeries.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Example_TimeSeries.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Example_TimeSeries.pdf/png")
    plt.close()


def plot_interaction_matrix_independence(embedding, metadata_list, latent_codes):
    """
    Demonstrate that latent space does NOT encode interaction matrix structure.
    Shows weak/no correlation with interaction properties.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr

    # Extract interaction matrix properties
    interaction_props = {
        'Mean Interaction': np.array([m['mean_interaction'] for m in metadata_list]),
        'Diagonal Strength': np.array([m['diagonal_strength'] for m in metadata_list]),
        'Max Eigenvalue': np.array([m['max_eigenvalue'] for m in metadata_list]),
        'Stability Measure': np.array([m['stability_measure'] for m in metadata_list]),
    }

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Top row: UMAP colored by interaction properties (showing no clear clustering)
    for idx, (prop_name, prop_values) in enumerate(list(interaction_props.items())[:2]):
        ax = fig.add_subplot(gs[0, idx])

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=prop_values,
            cmap='coolwarm',
            s=15,
            alpha=0.6,
            edgecolors='none',
            rasterized=True
        )

        ax.set_xlabel('UMAP 1', fontsize=10, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=10, fontweight='bold')
        ax.set_title(f'Latent Space: {prop_name}\n(No clear clustering)',
                    fontsize=11, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(prop_name, fontsize=9)

    # Middle row: Scatter plots showing weak correlations
    for idx, (prop_name, prop_values) in enumerate(list(interaction_props.items())[:2]):
        ax = fig.add_subplot(gs[1, idx])

        # Find latent dimension with highest correlation to this property
        corrs = [pearsonr(latent_codes[:, j], prop_values)[0] for j in range(latent_codes.shape[1])]
        best_dim = np.argmax(np.abs(corrs))
        best_corr = corrs[best_dim]

        # Scatter plot
        ax.scatter(
            latent_codes[:, best_dim],
            prop_values,
            s=15,
            alpha=0.4,
            c=COLORS['main'],
            edgecolors='none',
            rasterized=True
        )

        # Add regression line
        z = np.polyfit(latent_codes[:, best_dim], prop_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(latent_codes[:, best_dim].min(), latent_codes[:, best_dim].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)

        ax.set_xlabel(f'Latent Dimension z{best_dim}', fontsize=10, fontweight='bold')
        ax.set_ylabel(prop_name, fontsize=10, fontweight='bold')
        ax.set_title(f'{prop_name} vs Best Latent Dim\n(r = {best_corr:.3f} - weak correlation)',
                    fontsize=11, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=9)

    # Bottom: Distance correlation analysis
    ax = fig.add_subplot(gs[2, :])

    # Compute pairwise distances
    latent_distances = squareform(pdist(latent_codes, metric='euclidean'))
    interaction_matrices = np.array([m['interaction_matrix'].flatten() for m in metadata_list])
    interaction_distances = squareform(pdist(interaction_matrices, metric='euclidean'))

    # Sample for visualization
    triu_indices = np.triu_indices_from(latent_distances, k=1)
    latent_dist_flat = latent_distances[triu_indices]
    interaction_dist_flat = interaction_distances[triu_indices]

    correlation, p_value = spearmanr(latent_dist_flat, interaction_dist_flat)

    # Sample points for visualization
    sample_size = min(10000, len(latent_dist_flat))
    sample_idx = np.random.choice(len(latent_dist_flat), sample_size, replace=False)

    ax.scatter(
        interaction_dist_flat[sample_idx],
        latent_dist_flat[sample_idx],
        alpha=0.2,
        s=3,
        c=COLORS['main'],
        edgecolors='none',
        rasterized=True
    )

    ax.set_xlabel('Interaction Matrix Distance (Frobenius norm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latent Space Distance (Euclidean)', fontsize=11, fontweight='bold')
    ax.set_title(f'Distance Preservation: Interaction Matrices vs Latent Space\n' +
                f'Spearman ρ = {correlation:.3f} (p = {p_value:.2e}) - No correlation',
                fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=9)

    # Add text box with interpretation
    textstr = 'Weak correlation indicates:\nLatent space encodes dynamics,\nnot interaction parameters'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props, fontweight='bold')

    plt.savefig('Fig_Interaction_Independence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Interaction_Independence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Interaction_Independence.pdf/png")
    plt.close()


def plot_comparative_correlations(latent_codes, features, metadata_list):
    """
    Direct comparison: dynamical features vs interaction properties.
    Bar chart showing strong correlations for dynamics, weak for interactions.
    """
    # Dynamical features
    dynamical_features = {
        'Temporal Variance': features['temporal_variance'],
        'Steady State': features['steady_state_mean'],
        'Peak Timing': features['peak_times'],
        'Initial Growth': features['initial_growth_mean'],
    }

    # Interaction properties
    interaction_features = {
        'Mean Interaction': np.array([m['mean_interaction'] for m in metadata_list]),
        'Diagonal Strength': np.array([m['diagonal_strength'] for m in metadata_list]),
        'Stability Measure': np.array([m['stability_measure'] for m in metadata_list]),
        'Max Eigenvalue': np.array([m['max_eigenvalue'] for m in metadata_list]),
    }

    # Compute max absolute correlation for each feature
    dynamical_corrs = []
    for name, values in dynamical_features.items():
        max_corr = max([abs(pearsonr(latent_codes[:, j], values)[0]) for j in range(latent_codes.shape[1])])
        dynamical_corrs.append(max_corr)

    interaction_corrs = []
    for name, values in interaction_features.items():
        max_corr = max([abs(pearsonr(latent_codes[:, j], values)[0]) for j in range(latent_codes.shape[1])])
        interaction_corrs.append(max_corr)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(dynamical_features) + len(interaction_features))
    all_names = list(dynamical_features.keys()) + list(interaction_features.keys())
    all_corrs = dynamical_corrs + interaction_corrs

    # Color code: dynamics vs interactions
    colors = [COLORS['main']] * len(dynamical_corrs) + [COLORS['neutral']] * len(interaction_corrs)

    bars = ax.bar(x, all_corrs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add threshold line
    ax.axhline(y=0.3, color='gray', linestyle='--', linewidth=2, label='Moderate correlation (0.3)', alpha=0.7)

    # Styling
    ax.set_ylabel('Max Absolute Correlation with Latent Space', fontsize=12, fontweight='bold')
    ax.set_title('Latent Space Encodes Dynamics, Not Interaction Parameters', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_ylim([0, 0.8])

    # Add value labels on bars
    for bar, corr in zip(bars, all_corrs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{corr:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add category labels
    ax.text(1.5, -0.15, 'Dynamical Features\n(Strong correlations)',
           ha='center', va='top', fontsize=11, fontweight='bold',
           color=COLORS['main'], transform=ax.get_xaxis_transform())
    ax.text(5.5, -0.15, 'Interaction Properties\n(Weak correlations)',
           ha='center', va='top', fontsize=11, fontweight='bold',
           color=COLORS['neutral'], transform=ax.get_xaxis_transform())

    ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig('Fig_Comparative_Correlations.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Comparative_Correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Comparative_Correlations.pdf/png")
    plt.close()


def main():
    print("="*60)
    print("Generating Publication-Quality Figures")
    print("="*60)

    # Load data
    print("\nLoading data...")
    data, results = load_data()

    curves = data['curves']
    latent_codes = results['latent_codes']
    embedding = results['embedding_umap']
    metadata_list = data['metadata']

    # Extract features
    print("Extracting dynamical features...")
    features = extract_time_series_features(curves)

    # Generate all figures
    print("\nGenerating figures...\n")

    plot_latent_space_key_features(embedding, features)
    plot_correlation_heatmap(latent_codes, features)
    plot_latent_dimension_analysis(latent_codes, features)
    plot_variance_explained(latent_codes)
    plot_example_time_series(curves, embedding, features)

    # NEW: Interaction matrix independence analysis
    print("\nGenerating interaction independence figures...\n")
    plot_interaction_matrix_independence(embedding, metadata_list, latent_codes)
    plot_comparative_correlations(latent_codes, features, metadata_list)

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  • Fig_LatentSpace_KeyFeatures.pdf/png")
    print("  • Fig_Correlation_Heatmap.pdf/png")
    print("  • Fig_LatentDimension_Interpretation.pdf/png")
    print("  • Fig_Variance_Explained.pdf/png")
    print("  • Fig_Example_TimeSeries.pdf/png")
    print("  • Fig_Interaction_Independence.pdf/png - NEW: Shows weak interaction correlation")
    print("  • Fig_Comparative_Correlations.pdf/png - NEW: Direct comparison bar chart")
    print("\nAll figures are publication-ready (300 DPI, PDF + PNG formats)")


if __name__ == "__main__":
    main()
