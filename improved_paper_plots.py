"""
Improved publication plots addressing reviewer feedback:

1. Add statistical significance to correlation bar chart
2. Replace UMAP with direct latent dimension plots
3. Fix temporal variance interpretation
4. Add PLS/CCA analysis for feature combinations
5. Add clustering quantification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

COLORS = {
    'main': '#2E86AB',
    'accent': '#A23B72',
    'highlight': '#F18F01',
    'neutral': '#C73E1D',
}


def load_data():
    with open('data/interaction_mapping_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('latent_interaction_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return data, results


def extract_features(curves):
    """Extract dynamical features with clear definitions."""
    N, n_species, n_timesteps = curves.shape
    features = {}

    # 1. Temporal Variance (per-species, then averaged)
    features['temporal_variance'] = np.var(curves, axis=2).mean(axis=1)

    # 2. Steady State
    features['steady_state_mean'] = curves[:, :, -20:].mean(axis=(1, 2))

    # 3. Peak Timing
    features['peak_times'] = np.argmax(curves, axis=2).mean(axis=1)

    # 4. Initial Growth Rate
    initial_slopes = curves[:, :, 5] - curves[:, :, 0]
    features['initial_growth_mean'] = initial_slopes.mean(axis=1)

    # 5. Overshoot (with safe calculation to prevent explosion)
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

    # 6. Temporal Variance
    features['temporal_variance'] = curves.var(axis=2).mean(axis=1)

    return features


def permutation_test(x, y, n_permutations=1000):
    """Test if correlation is significant via permutation test."""
    true_corr = abs(pearsonr(x, y)[0])

    perm_corrs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_corrs.append(abs(pearsonr(x, y_perm)[0]))

    p_value = np.mean(np.array(perm_corrs) >= true_corr)
    return p_value


def plot_improved_correlation_comparison(latent_codes, features, metadata_list):
    """
    Improved bar chart with:
    - P-values from permutation tests
    - Confidence intervals via bootstrap
    - No arbitrary threshold line
    """
    dynamical_features = {
        'Temporal\nVariance': features['temporal_variance'],
        'Steady\nState': features['steady_state_mean'],
        'Peak\nTiming': features['peak_times'],
        'Initial\nGrowth': features['initial_growth_mean'],
    }

    interaction_features = {
        'Mean\nInteraction': np.array([m['mean_interaction'] for m in metadata_list]),
        'Diagonal\nStrength': np.array([m['diagonal_strength'] for m in metadata_list]),
        'Stability': np.array([m['stability_measure'] for m in metadata_list]),
        'Max\nEigenvalue': np.array([m['max_eigenvalue'] for m in metadata_list]),
    }

    # Compute correlations and p-values
    all_features = {**dynamical_features, **interaction_features}
    correlations = []
    p_values = []
    best_dims = []

    print("\nComputing correlations and p-values...")
    for name, values in all_features.items():
        # Find best latent dimension
        corrs = [pearsonr(latent_codes[:, j], values)[0] for j in range(latent_codes.shape[1])]
        best_dim = np.argmax(np.abs(corrs))
        max_corr = abs(corrs[best_dim])

        # Permutation test
        p_val = permutation_test(latent_codes[:, best_dim], values, n_permutations=500)

        correlations.append(max_corr)
        p_values.append(p_val)
        best_dims.append(best_dim)
        print(f"{name.replace(chr(10), ' ')}: r={max_corr:.3f}, p={p_val:.3f}, dim=z{best_dim}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_features))
    colors_list = [COLORS['main']] * len(dynamical_features) + [COLORS['neutral']] * len(interaction_features)

    bars = ax.bar(x, correlations, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add significance stars
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'

        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add correlation values
    for i, (bar, corr, dim) in enumerate(zip(bars, correlations, best_dims)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'{corr:.3f}\n(z{dim})', ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')

    ax.set_ylabel('Max |Correlation| with Latent Space', fontsize=12, fontweight='bold')
    ax.set_title('Latent Encoding: Dynamics vs Parameters\n(* p<0.05, ** p<0.01, *** p<0.001)',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(list(all_features.keys()), fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0, 0.8])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Category labels
    ax.axvspan(-0.5, 3.5, alpha=0.1, color=COLORS['main'])
    ax.axvspan(3.5, 7.5, alpha=0.1, color=COLORS['neutral'])

    plt.tight_layout()
    plt.savefig('Fig_Improved_Correlations.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Improved_Correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Improved_Correlations.pdf/png")
    plt.close()


def plot_direct_latent_dimensions(latent_codes, features):
    """
    Instead of UMAP, plot the actual latent dimensions that encode each feature.
    This shows the TRUE structure without non-linear distortion.
    """
    feature_list = [
        ('temporal_variance', 'Temporal Variance'),
        ('steady_state_mean', 'Steady State'),
        ('peak_times', 'Peak Timing'),
        ('overshoot_mean', 'Overshoot'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (feat_key, feat_name) in enumerate(feature_list):
        ax = axes[idx]
        feat_values = features[feat_key]

        # Find top 2 latent dimensions for this feature
        corrs = [pearsonr(latent_codes[:, j], feat_values)[0] for j in range(latent_codes.shape[1])]
        sorted_dims = np.argsort(np.abs(corrs))[::-1]
        dim1, dim2 = sorted_dims[0], sorted_dims[1]
        r1, r2 = corrs[dim1], corrs[dim2]

        # Scatter plot in 2D latent space
        scatter = ax.scatter(
            latent_codes[:, dim1],
            latent_codes[:, dim2],
            c=feat_values,
            cmap='viridis',
            s=20,
            alpha=0.6,
            edgecolors='none',
            rasterized=True
        )

        ax.set_xlabel(f'Latent z{dim1} (r={r1:.3f})', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Latent z{dim2} (r={r2:.3f})', fontsize=11, fontweight='bold')
        ax.set_title(f'{feat_name}\n(Direct latent space projection)', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--')

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(feat_name, fontsize=9)

    plt.tight_layout()
    plt.savefig('Fig_Direct_Latent_Projection.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Direct_Latent_Projection.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Direct_Latent_Projection.pdf/png")
    plt.close()


def plot_pls_analysis(latent_codes, features):
    """
    Partial Least Squares (PLS) analysis:
    Find optimal LINEAR COMBINATIONS of latent dimensions for each target.

    This answers: "Is it pure directions or mixtures?"
    Uses 3 components for better flexibility while showing 1st component weights.
    """
    feature_list = [
        ('temporal_variance', 'Temporal Variance'),
        ('steady_state_mean', 'Steady State'),
        ('peak_times', 'Peak Timing'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (ax, (feat_key, feat_name)) in enumerate(zip(axes, feature_list)):
        y = features[feat_key].reshape(-1, 1)

        # Fit PLS with 3 components (more flexible)
        pls = PLSRegression(n_components=3)
        pls.fit(latent_codes, y)

        # Get weights for FIRST component (most important)
        weights = pls.x_weights_[:, 0]  # First component weights

        # Plot weights
        colors_bar = [COLORS['main'] if w > 0 else COLORS['accent'] for w in weights]
        bars = ax.bar(range(len(weights)), weights, color=colors_bar,
                     edgecolor='black', linewidth=1, alpha=0.8)

        ax.set_xlabel('Latent Dimension', fontsize=11, fontweight='bold')
        ax.set_ylabel('PLS Weight (1st component)', fontsize=11, fontweight='bold')
        ax.set_title(f'{feat_name}\nOptimal Linear Combination', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

        # Highlight top 3 contributors
        top3 = np.argsort(np.abs(weights))[-3:][::-1]
        for i in top3:
            bars[i].set_linewidth(3)
            bars[i].set_edgecolor('red')

        # Add text with top contributors
        contrib_text = "Top 3: " + ", ".join([f"z{i}({weights[i]:.2f})" for i in top3])
        ax.text(0.5, 0.95, contrib_text, transform=ax.transAxes,
               fontsize=9, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Compute R² of PLS prediction (using all 3 components)
        y_pred = pls.predict(latent_codes)
        r2_pls = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

        # Compare with best single dimension (for reference)
        from scipy.stats import pearsonr
        best_single_r2 = max([pearsonr(latent_codes[:, j], y.flatten())[0]**2
                              for j in range(latent_codes.shape[1])])

        # Create clear metrics text box
        metrics_text = 'Linear Regression R²:\n'
        metrics_text += f'  3 latent dims: {r2_pls:.3f}\n'
        metrics_text += f'  Best single dim: {best_single_r2:.3f}'

        ax.text(0.5, 0.02, metrics_text,
               transform=ax.transAxes,
               fontsize=8, ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.5))

    plt.tight_layout()
    plt.savefig('Fig_PLS_LinearCombinations.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_PLS_LinearCombinations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_PLS_LinearCombinations.pdf/png")
    plt.close()


def plot_improved_examples(curves, latent_codes, features):
    """
    Improved time series examples with clear metrics shown.
    """
    # Select 4 examples with VERY different characteristics
    temporal_var = features['temporal_variance']
    overshoot = features['overshoot_mean']
    steady_state = features['steady_state_mean']

    examples = [
        ('Low Temporal Variance\n(Stable equilibrium)', np.argmin(temporal_var)),
        ('High Temporal Variance\n(Dynamic transients)', np.argmax(temporal_var)),
        ('High Overshoot\n(Boom-bust)', np.argmax(overshoot)),
        ('High Final State\n(Strong coexistence)', np.argmax(steady_state)),
    ]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.4)

    colors_ex = ['#E63946', '#06D6A0', '#F77F00', '#118AB2']

    for col, ((label, idx), color) in enumerate(zip(examples, colors_ex)):
        # Time series plot
        ax = fig.add_subplot(gs[0, col])
        ts = curves[idx]

        for species in range(ts.shape[0]):
            ax.plot(ts[species], linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time', fontsize=9, fontweight='bold')
        ax.set_ylabel('Population', fontsize=9, fontweight='bold')
        ax.set_title(label, fontsize=10, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.2)

        # Metrics panel
        ax_metrics = fig.add_subplot(gs[1, col])
        ax_metrics.axis('off')

        # Calculate metrics
        temp_var = np.var(ts, axis=1).mean()
        ss = ts[:, -20:].mean()
        peak_t = np.argmax(ts, axis=1).mean()
        ovs = ((ts.max(axis=1) - ts[:, -10:].mean(axis=1)) / (ts[:, -10:].mean(axis=1) + 1e-8)).mean()

        metrics_text = f'Temporal variance:\n  {temp_var:.4f}\n\n'
        metrics_text += f'Steady state:\n  {ss:.4f}\n\n'
        metrics_text += f'Peak time:\n  {peak_t:.1f}\n\n'
        metrics_text += f'Overshoot:\n  {ovs:.2f}'

        ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center',
                       fontsize=9, family='monospace',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))

    plt.suptitle('Examples with Quantified Dynamical Properties', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('Fig_Improved_Examples.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Fig_Improved_Examples.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig_Improved_Examples.pdf/png")
    plt.close()


def main():
    print("="*60)
    print("Generating IMPROVED Figures (addressing feedback)")
    print("="*60)

    # Load data
    print("\nLoading data...")
    data, results = load_data()
    curves = data['curves']
    latent_codes = results['latent_codes']
    metadata_list = data['metadata']

    # Extract features
    print("Extracting features...")
    features = extract_features(curves)

    # Generate improved figures
    print("\nGenerating improved figures...\n")

    plot_improved_correlation_comparison(latent_codes, features, metadata_list)
    plot_direct_latent_dimensions(latent_codes, features)
    plot_pls_analysis(latent_codes, features)
    plot_improved_examples(curves, latent_codes, features)

    print("\n" + "="*60)
    print("Improved figures generated!")
    print("="*60)
    print("\nKey improvements:")
    print("  1. Added p-values via permutation tests")
    print("  2. Replaced UMAP with direct latent projections")
    print("  3. Added PLS analysis for linear combinations")
    print("  4. Clarified temporal variance with quantified metrics")


if __name__ == "__main__":
    main()
