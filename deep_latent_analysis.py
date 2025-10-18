"""
DEEP LATENT SPACE SEMANTICS ANALYSIS

Goes beyond surface-level metrics to understand:
- WHAT is the latent space encoding?
- WHY does it work so well?
- HOW can we interpret individual dimensions?

Generates advanced visualizations revealing the semantic structure
and causal relationships in the learned representations.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import seaborn as sns

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
})

MODEL_PATH = 'model_ckpts/model_final_30.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_DIR = Path("deep_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_and_encode():
    """Load everything and encode."""
    print("Loading data and encoding...")

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    curves = torch.tensor(data['curves'], dtype=torch.float32)
    metadata = data['metadata']

    # Load model
    model = LSTM_VAE(config=model_config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Encode and reconstruct
    latent_codes = []
    reconstructions = []
    scales = []
    with torch.no_grad():
        batch_size = 64
        for i in tqdm(range(0, len(curves), batch_size), desc="Encoding"):
            batch = curves[i:i+batch_size].to(DEVICE)
            # Get full reconstruction
            X_hat, mu, log_var, max_vals_pred, _ = model(batch, teacher_forcing_ratio=0.0)

            # Apply scale to reconstruction
            X_hat_scaled = X_hat * max_vals_pred.unsqueeze(-1)  # (N, 7, L) * (N, 7, 1) = (N, 7, L)

            latent_codes.append(mu.cpu())
            reconstructions.append(X_hat_scaled.cpu())
            scales.append(max_vals_pred.cpu())

    latent_codes = torch.cat(latent_codes, dim=0).numpy()
    reconstructions = torch.cat(reconstructions, dim=0)
    scales = torch.cat(scales, dim=0)

    # Unnormalize curves using max_values
    print("\nUnnormalizing curves...")
    max_vals = data['max_values']  # Shape: (N, 7)
    unnormalized_curves = curves * torch.tensor(max_vals, dtype=torch.float32).unsqueeze(-1)
    print(f"Unnormalized curves shape: {unnormalized_curves.shape}")
    print(f"Sample unnormalized max values: {unnormalized_curves[0].max(dim=1)[0][:3]}")

    # Extract all possible features
    features = {}

    # Dynamical features from UNNORMALIZED curves
    for i in tqdm(range(len(unnormalized_curves)), desc="Extracting features"):
        curve = unnormalized_curves[i].numpy()

        # Peak analysis
        peak_times = np.argmax(curve, axis=1)
        peak_values = np.max(curve, axis=1)

        # Steady state
        steady = np.mean(curve[:, -10:], axis=1)

        if i == 0:
            features['peak_time_mean'] = []
            features['peak_time_std'] = []
            features['peak_value_mean'] = []
            features['steady_state_sum'] = []
            features['steady_state_max'] = []
            features['diversity'] = []
            features['dominance'] = []
            features['n_survivors'] = []
            features['temporal_var'] = []
            features['initial_growth'] = []

        features['peak_time_mean'].append(np.mean(peak_times))
        features['peak_time_std'].append(np.std(peak_times))
        features['peak_value_mean'].append(np.mean(peak_values))
        features['steady_state_sum'].append(np.sum(steady))
        features['steady_state_max'].append(np.max(steady))

        # Diversity (Shannon entropy)
        steady_probs = steady / (np.sum(steady) + 1e-10)
        steady_probs = steady_probs[steady_probs > 1e-10]
        features['diversity'].append(-np.sum(steady_probs * np.log(steady_probs + 1e-10)))

        features['dominance'].append(np.max(steady) / (np.sum(steady) + 1e-10))
        features['n_survivors'].append(np.sum(steady > 0.01))
        features['temporal_var'].append(np.mean(np.var(curve, axis=1)))
        features['initial_growth'].append(np.mean(np.gradient(curve[:, :10], axis=1)))

    # Convert to arrays
    for key in features.keys():
        features[key] = np.array(features[key])

    # GLV parameters from metadata
    glv_params = {
        'max_eigenvalue': np.array([m['max_eigenvalue'] for m in metadata]),
        'mean_interaction': np.array([m['mean_interaction'] for m in metadata]),
        'interaction_std': np.array([m['interaction_std'] for m in metadata]),
        'diagonal_strength': np.array([m['diagonal_strength'] for m in metadata]),
        'stability_measure': np.array([m['stability_measure'] for m in metadata]),
    }

    # Growth rates
    growth_stats = {
        'growth_mean': np.array([np.mean(m['growth_rates']) for m in metadata]),
        'growth_std': np.array([np.std(m['growth_rates']) for m in metadata]),
        'growth_max': np.array([np.max(m['growth_rates']) for m in metadata]),
    }

    # Steady state from GLV
    steady_glv = {
        'steady_glv_sum': np.array([np.sum(m['steady_state']) for m in metadata]),
        'steady_glv_max': np.array([np.max(m['steady_state']) for m in metadata]),
        'steady_glv_min': np.array([np.min(m['steady_state']) for m in metadata]),
    }

    all_features = {**features, **glv_params, **growth_stats, **steady_glv}

    # Debug: check scaling
    print(f"\n[DEBUG] Original curves channel 0 max: {curves[:, 0, :].max():.4f}")
    print(f"[DEBUG] Reconstructions channel 0 max: {reconstructions[:, 0, :].max():.4f}")
    print(f"[DEBUG] Scales channel 0: min={scales[:, 0].min():.4f}, max={scales[:, 0].max():.4f}")
    print(f"[DEBUG] Sample reconstruction maxes: {reconstructions[0, :, :].max(dim=1)[0][:3]}")

    return {
        'model': model,
        'latent_codes': latent_codes,
        'features': all_features,
        'metadata': metadata,
        'curves': unnormalized_curves,  # Using ORIGINAL unnormalized curves
        'reconstructions': reconstructions,  # Keep reconstructions for comparison
        'scales': scales,
    }


def figure7_glv_parameter_encoding(data):
    """
    Figure 7: What GLV parameters are encoded in latent space?

    Shows which interaction matrix properties, eigenvalues, growth rates
    are captured by the latent representation.
    """
    print("\n" + "="*70)
    print("FIGURE 7: GLV PARAMETER ENCODING")
    print("="*70)

    latent = data['latent_codes']
    features = data['features']

    # Key GLV parameters to analyze
    glv_features = [
        'max_eigenvalue', 'mean_interaction', 'interaction_std',
        'diagonal_strength', 'stability_measure',
        'growth_mean', 'growth_std', 'growth_max',
    ]

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # For each GLV parameter, find best encoding dimension and show relationship
    results = []

    for idx, param in enumerate(glv_features):
        row = idx // 4
        col = idx % 4

        ax = fig.add_subplot(gs[row, col])

        # Find best latent dimension
        correlations = []
        for dim in range(latent.shape[1]):
            r = np.corrcoef(latent[:, dim], features[param])[0, 1]
            correlations.append(abs(r))

        best_dim = np.argmax(correlations)
        best_corr = np.corrcoef(latent[:, best_dim], features[param])[0, 1]

        results.append({
            'param': param,
            'best_dim': best_dim,
            'correlation': best_corr
        })

        # Scatter plot with hexbin
        ax.hexbin(latent[:, best_dim], features[param],
                 gridsize=25, cmap='Blues', mincnt=1)

        # Fit line
        z = np.polyfit(latent[:, best_dim], features[param], 1)
        p = np.poly1d(z)
        x_line = np.linspace(latent[:, best_dim].min(), latent[:, best_dim].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)

        # Labels
        param_nice = param.replace('_', ' ').title()
        ax.set_xlabel(f'Latent Dim {best_dim}', fontweight='bold')
        ax.set_ylabel(param_nice, fontweight='bold')
        ax.set_title(f'{param_nice}\nr = {best_corr:.3f}', fontweight='bold', fontsize=11)
        ax.grid(alpha=0.2)

    # Panel with random forest feature importance
    ax_rf = fig.add_subplot(gs[2, :2])

    print("Computing Random Forest importance...")
    X = latent
    importances_all = []

    for param in glv_features:
        y = features[param]
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
        rf.fit(X, y)
        importances_all.append(rf.feature_importances_)

    importances_all = np.array(importances_all)
    mean_importance = importances_all.mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_importance)[::-1][:15]

    colors = plt.cm.viridis(mean_importance[sorted_idx] / mean_importance[sorted_idx].max())
    bars = ax_rf.barh(range(len(sorted_idx)), mean_importance[sorted_idx], color=colors, edgecolor='black')
    ax_rf.set_yticks(range(len(sorted_idx)))
    ax_rf.set_yticklabels([f'Dim {i}' for i in sorted_idx])
    ax_rf.set_xlabel('Mean Importance (RF)', fontweight='bold')
    ax_rf.set_title('Top 15 Dimensions by GLV Parameter Encoding', fontweight='bold', fontsize=12)
    ax_rf.invert_yaxis()
    ax_rf.grid(alpha=0.2, axis='x')

    # Panel with encoding summary table
    ax_table = fig.add_subplot(gs[2, 2:])
    ax_table.axis('off')

    # Sort results by correlation strength
    results_sorted = sorted(results, key=lambda x: abs(x['correlation']), reverse=True)

    table_text = "GLV Parameter Encoding Summary\n" + "="*55 + "\n"
    table_text += f"{'Parameter':<25} {'Best Dim':<10} {'r':<10}\n"
    table_text += "-"*55 + "\n"

    for r in results_sorted:
        param_nice = r['param'].replace('_', ' ').title()
        table_text += f"{param_nice:<25} {r['best_dim']:<10} {r['correlation']:>+.3f}\n"

    table_text += "\n" + "="*55 + "\n"
    table_text += f"Mean |r|: {np.mean([abs(r['correlation']) for r in results]):.3f}\n"
    table_text += f"Strong (|r|>0.4): {sum([abs(r['correlation']) > 0.4 for r in results])}/{len(results)}\n"

    ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes,
                 fontsize=9, family='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.savefig(OUTPUT_DIR / 'Figure7_GLV_Parameter_Encoding.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure7_GLV_Parameter_Encoding.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 7")
    plt.close()


def figure8_dimension_semantics(data):
    """
    Figure 8: Semantic interpretation of latent dimensions.

    For top dimensions, show:
    - What they correlate with most strongly
    - Visual examples of traversals
    - Perturbation sensitivity
    - Hierarchical importance
    """
    print("\n" + "="*70)
    print("FIGURE 8: DIMENSION SEMANTICS & INTERPRETATION")
    print("="*70)

    model = data['model']
    latent = data['latent_codes']
    features = data['features']

    # Compute variance per dimension (activity)
    dim_variance = np.var(latent, axis=0)

    # Compute max absolute correlation per dimension
    all_feature_names = list(features.keys())
    max_corr_per_dim = []
    best_feature_per_dim = []

    for dim in range(latent.shape[1]):
        corrs = [abs(np.corrcoef(latent[:, dim], features[f])[0, 1])
                for f in all_feature_names]
        max_corr_per_dim.append(max(corrs))
        best_feature_per_dim.append(all_feature_names[np.argmax(corrs)])

    # Rank dimensions by "importance" (variance * correlation)
    importance = dim_variance * np.array(max_corr_per_dim)
    top_dims = np.argsort(importance)[::-1][:10]

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(5, 6, figure=fig, hspace=0.4, wspace=0.4)

    # Panel A: Dimension importance ranking
    ax_importance = fig.add_subplot(gs[0, :3])

    colors = plt.cm.RdYlGn(importance[top_dims] / importance[top_dims].max())
    bars = ax_importance.barh(range(len(top_dims)), importance[top_dims], color=colors, edgecolor='black')
    ax_importance.set_yticks(range(len(top_dims)))
    ax_importance.set_yticklabels([f'Dim {d}' for d in top_dims])
    ax_importance.set_xlabel('Importance (Variance × Correlation)', fontweight='bold')
    ax_importance.set_title('Top 10 Most Important Dimensions', fontweight='bold', fontsize=13)
    ax_importance.invert_yaxis()
    ax_importance.grid(alpha=0.2, axis='x')

    # Add feature labels
    for i, dim in enumerate(top_dims):
        feat = best_feature_per_dim[dim].replace('_', ' ')
        ax_importance.text(importance[dim], i, f'  {feat}',
                          va='center', fontsize=8, fontweight='bold')

    # Panel B: Semantic meaning table
    ax_semantic = fig.add_subplot(gs[0, 3:])
    ax_semantic.axis('off')

    semantic_text = "Dimension Semantics (Top 10)\n" + "="*70 + "\n"
    semantic_text += f"{'Dim':<5} {'Primary Feature':<25} {'r':<8} {'Variance':<10}\n"
    semantic_text += "-"*70 + "\n"

    for dim in top_dims:
        feat = best_feature_per_dim[dim].replace('_', ' ').title()
        r = max_corr_per_dim[dim]
        var = dim_variance[dim]
        semantic_text += f"{dim:<5} {feat:<25} {r:>+.3f}   {var:>8.3f}\n"

    ax_semantic.text(0.02, 0.98, semantic_text, transform=ax_semantic.transAxes,
                    fontsize=9, family='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panels C-E: Traversals for top 3 dimensions
    print("\nGenerating traversals for top dimensions...")

    base_idx = np.random.randint(0, len(latent))
    base_z = torch.FloatTensor(latent[base_idx:base_idx+1])

    for panel_idx, dim in enumerate(top_dims[:3]):
        row = 1 + panel_idx

        # Get feature name
        feat_name = best_feature_per_dim[dim].replace('_', ' ').title()

        # Traverse dimension
        std = np.std(latent[:, dim])
        values = np.linspace(-2*std, 2*std, 6)

        for col_offset, val in enumerate(values):
            ax = fig.add_subplot(gs[row, col_offset])

            z_mod = base_z.clone()
            z_mod[0, dim] = float(val)

            with torch.no_grad():
                decoded, _ = model.decode(z_mod.to(DEVICE))

            curve = decoded[0].cpu().numpy()

            # Plot 3 representative species
            for species in [0, 2, 4]:
                ax.plot(curve[species], alpha=0.8, linewidth=2, color=f'C{species}')

            ax.set_ylim(-0.05, 1.1)
            ax.set_title(f'{val/std:.1f}σ', fontsize=10)

            if col_offset == 0:
                ax.set_ylabel(f'Dim {dim}\n({feat_name})', fontweight='bold', fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([0, 0.5, 1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Panel F: Correlation heatmap for top dims vs all features
    ax_heatmap = fig.add_subplot(gs[3:, :3])

    # Compute correlation matrix
    corr_matrix = np.zeros((len(top_dims), len(all_feature_names)))
    for i, dim in enumerate(top_dims):
        for j, feat in enumerate(all_feature_names):
            # Check for zero variance
            if np.std(latent[:, dim]) < 1e-10 or np.std(features[feat]) < 1e-10:
                corr_matrix[i, j] = 0.0
            else:
                r = np.corrcoef(latent[:, dim], features[feat])[0, 1]
                corr_matrix[i, j] = 0 if (np.isnan(r) or np.isinf(r)) else r

    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    im = ax_heatmap.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax_heatmap.set_yticks(range(len(top_dims)))
    ax_heatmap.set_yticklabels([f'Dim {d}' for d in top_dims], fontsize=9)
    ax_heatmap.set_xticks(range(len(all_feature_names)))
    ax_heatmap.set_xticklabels([f.replace('_', ' ')[:15] for f in all_feature_names],
                               rotation=45, ha='right', fontsize=8)
    ax_heatmap.set_title('Top Dimensions vs All Features', fontweight='bold', fontsize=12)

    plt.colorbar(im, ax=ax_heatmap, label='Pearson r')

    # Panel G: Hierarchical clustering of dimensions
    ax_dendro = fig.add_subplot(gs[3:, 3:])

    print("Computing hierarchical clustering...")
    # Cluster dimensions by their correlation profiles
    corr_profiles = np.zeros((latent.shape[1], len(all_feature_names)))
    for i in range(latent.shape[1]):
        for j, feat in enumerate(all_feature_names):
            # Check for zero variance
            if np.std(latent[:, i]) < 1e-10 or np.std(features[feat]) < 1e-10:
                corr_profiles[i, j] = 0.0
            else:
                r = np.corrcoef(latent[:, i], features[feat])[0, 1]
                corr_profiles[i, j] = 0 if (np.isnan(r) or np.isinf(r)) else r

    # Replace any remaining NaNs and infs
    corr_profiles = np.nan_to_num(corr_profiles, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure all values are finite
    if not np.all(np.isfinite(corr_profiles)):
        print("Warning: Non-finite values in correlation profiles, replacing with zeros")
        corr_profiles = np.where(np.isfinite(corr_profiles), corr_profiles, 0.0)

    # Use 'average' linkage which is more robust than 'ward'
    linkage_matrix = linkage(corr_profiles, method='average')
    dendro = dendrogram(linkage_matrix, ax=ax_dendro, orientation='right',
                       labels=[f'D{i}' for i in range(latent.shape[1])],
                       leaf_font_size=8, color_threshold=10)

    ax_dendro.set_xlabel('Distance', fontweight='bold')
    ax_dendro.set_title('Hierarchical Clustering\n(by correlation profiles)',
                       fontweight='bold', fontsize=12)
    ax_dendro.grid(alpha=0.2, axis='x')

    plt.savefig(OUTPUT_DIR / 'Figure8_Dimension_Semantics.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure8_Dimension_Semantics.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 8")
    plt.close()


def figure9_dynamical_regimes(data):
    """
    Figure 9: Dynamical regimes in latent space.

    Shows how different ecological behaviors cluster in latent space.
    """
    print("\n" + "="*70)
    print("FIGURE 9: DYNAMICAL REGIMES")
    print("="*70)

    latent = data['latent_codes']
    features = data['features']
    curves = data['curves']
    model = data['model']

    # Define regimes based on dynamics
    regimes = np.zeros(len(latent), dtype=int)

    # Regime classification
    # 0: Low diversity, stable
    # 1: High diversity, stable
    # 2: Dominant species
    # 3: Oscillatory/unstable

    diversity = features['diversity']
    dominance = features['dominance']
    temporal_var = features['temporal_var']

    # Use data-driven thresholds (medians)
    div_thresh = np.median(diversity)
    dom_thresh = np.median(dominance)
    var_thresh = np.median(temporal_var)

    print(f"Thresholds: diversity={div_thresh:.3f}, dominance={dom_thresh:.3f}, temporal_var={var_thresh:.4f}")

    for i in range(len(latent)):
        if diversity[i] < div_thresh:
            if dominance[i] > dom_thresh:
                regimes[i] = 2  # Dominant
            else:
                regimes[i] = 0  # Low diversity stable
        else:
            if temporal_var[i] > var_thresh:
                regimes[i] = 3  # Oscillatory
            else:
                regimes[i] = 1  # High diversity stable

    regime_names = ['Low Div Stable', 'High Div Stable', 'Dominant', 'Oscillatory']
    regime_colors = ['#377eb8', '#4daf4a', '#e41a1c', '#ff7f00']

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.35)

    # Panel A: PCA colored by regime
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(latent)

    ax_pca = fig.add_subplot(gs[0, 0])

    for regime_id, (name, color) in enumerate(zip(regime_names, regime_colors)):
        mask = regimes == regime_id
        ax_pca.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                      c=color, label=name, s=20, alpha=0.6, edgecolors='black', linewidth=0.3)

    ax_pca.set_xlabel('PC 1', fontweight='bold')
    ax_pca.set_ylabel('PC 2', fontweight='bold')
    ax_pca.set_title('Dynamical Regimes (PCA)', fontweight='bold', fontsize=12)
    ax_pca.legend(loc='best', fontsize=9)
    ax_pca.grid(alpha=0.2)

    # Panel B: t-SNE colored by regime
    from sklearn.manifold import TSNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)

    # Subsample for speed
    n_samples = min(2000, len(latent))
    idx = np.random.choice(len(latent), n_samples, replace=False)
    tsne_coords = tsne.fit_transform(latent[idx])

    ax_tsne = fig.add_subplot(gs[0, 1])

    for regime_id, (name, color) in enumerate(zip(regime_names, regime_colors)):
        mask = regimes[idx] == regime_id
        ax_tsne.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                       c=color, label=name, s=20, alpha=0.6, edgecolors='black', linewidth=0.3)

    ax_tsne.set_xlabel('t-SNE 1', fontweight='bold')
    ax_tsne.set_ylabel('t-SNE 2', fontweight='bold')
    ax_tsne.set_title('Dynamical Regimes (t-SNE)', fontweight='bold', fontsize=12)
    ax_tsne.legend(loc='best', fontsize=9)
    ax_tsne.grid(alpha=0.2)

    # Panel C: Regime statistics
    ax_stats = fig.add_subplot(gs[0, 2:])
    ax_stats.axis('off')

    stats_text = "Regime Statistics\n" + "="*60 + "\n"
    stats_text += f"{'Regime':<20} {'Count':<10} {'%':<10}\n"
    stats_text += "-"*60 + "\n"

    for regime_id, name in enumerate(regime_names):
        count = np.sum(regimes == regime_id)
        pct = 100 * count / len(regimes)
        stats_text += f"{name:<20} {count:<10} {pct:>6.1f}%\n"

    stats_text += "\n" + "="*60 + "\n"
    stats_text += "Feature Means by Regime:\n"
    stats_text += f"{'Regime':<20} {'Div':<8} {'Dom':<8} {'TVar':<8}\n"
    stats_text += "-"*60 + "\n"

    for regime_id, name in enumerate(regime_names):
        mask = regimes == regime_id
        if np.sum(mask) == 0:
            # No samples in this regime
            stats_text += f"{name:<20} {'---':>6}  {'---':>6}  {'---':>6}\n"
        else:
            div_mean = np.nanmean(diversity[mask])
            dom_mean = np.nanmean(dominance[mask])
            var_mean = np.nanmean(temporal_var[mask])
            stats_text += f"{name:<20} {div_mean:>6.2f}  {dom_mean:>6.2f}  {var_mean:>6.3f}\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, family='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Define distinct colors for each species
    species_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

    # Panels D-G: Representative examples from each regime
    for regime_id, (name, color) in enumerate(zip(regime_names, regime_colors)):
        ax = fig.add_subplot(gs[1, regime_id])

        # Find all samples in this regime
        mask = regimes == regime_id

        if np.sum(mask) > 0:
            # Find sample closest to regime centroid in latent space
            regime_latent = latent[mask]
            centroid = regime_latent.mean(axis=0)

            # Compute distances to centroid
            distances = np.linalg.norm(regime_latent - centroid, axis=1)

            # Get the single closest sample (most representative)
            closest_idx = np.argmin(distances)
            representative_idx = np.where(mask)[0][closest_idx]

            # Get the representative curve
            rep_curve = curves[representative_idx]  # (7, timesteps)

            # Debug: print statistics
            print(f"  Regime '{name}': showing most representative sample (#{representative_idx}/{np.sum(mask)} total)")
            rep_curve_np = rep_curve.numpy() if torch.is_tensor(rep_curve) else rep_curve
            print(f"    Channel 0 (red) max: {rep_curve_np[0].max():.4f}")
            print(f"    All channel maxes: {[rep_curve_np[i].max() for i in range(7)]}")

            # Plot all 7 species
            max_val = 0
            for species in range(7):
                curve_data = rep_curve_np[species]
                time = np.arange(len(curve_data))
                sp_color = species_colors[species]

                # Plot curve
                ax.plot(time, curve_data, linewidth=2.5, alpha=0.9, color=sp_color,
                       label=f'Sp {species+1}' if regime_id == 0 else '', zorder=10)
                max_val = max(max_val, curve_data.max())

            # Set y-limits dynamically based on data
            ax.set_ylim(-0.05, max(1.1, max_val * 1.1))
        else:
            ax.set_ylim(-0.05, 1.1)
        ax.set_title(name, fontweight='bold', fontsize=11, color=color)
        ax.set_xlabel('Time' if regime_id == 1 else '')
        ax.set_ylabel('Abundance' if regime_id == 0 else '')
        if regime_id == 0:
            ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / 'Figure9_Dynamical_Regimes.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure9_Dynamical_Regimes.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 9")
    plt.close()


def figure10_predictive_power(data):
    """
    Figure 10: What can we predict from latent codes?

    Shows the predictive power and information flow.
    """
    print("\n" + "="*70)
    print("FIGURE 10: PREDICTIVE POWER & INFORMATION")
    print("="*70)

    latent = data['latent_codes']
    features = data['features']

    # Train predictive models for all features
    feature_list = list(features.keys())

    # Linear vs non-linear comparison
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor

    linear_scores = []
    nonlinear_scores = []
    raw_linear_scores = []  # Store raw scores for debugging
    raw_nonlinear_scores = []

    print("Training predictive models...")

    X = latent
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_train = int(0.8 * len(X))
    X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]

    for feat in tqdm(feature_list, desc="Features"):
        y = features[feat]
        y_train, y_test = y[:n_train], y[n_train:]

        # Linear model
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(X_train, y_train)
        linear_r2 = linear_model.score(X_test, y_test)
        raw_linear_scores.append(linear_r2)
        linear_scores.append(max(0, linear_r2))  # Clip negative

        # Non-linear model
        nonlinear_model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                    random_state=RANDOM_SEED)
        nonlinear_model.fit(X_train, y_train)
        nonlinear_r2 = nonlinear_model.score(X_test, y_test)
        raw_nonlinear_scores.append(nonlinear_r2)
        nonlinear_scores.append(max(0, nonlinear_r2))

    # Print features with negative/low R²
    print("\nFeatures with R² < 0.05:")
    for i, feat in enumerate(feature_list):
        if linear_scores[i] < 0.05:
            print(f"  {feat}: Linear R²={raw_linear_scores[i]:.4f} (clipped to {linear_scores[i]:.4f}), "
                  f"Non-linear R²={raw_nonlinear_scores[i]:.4f} (clipped to {nonlinear_scores[i]:.4f})")

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Linear vs non-linear R²
    ax_compare = fig.add_subplot(gs[0, 0])

    x = np.arange(len(feature_list))
    width = 0.35

    ax_compare.barh(x - width/2, linear_scores, width, label='Linear (Ridge)',
                   color='steelblue', alpha=0.8, edgecolor='black')
    ax_compare.barh(x + width/2, nonlinear_scores, width, label='Non-linear (GBT)',
                   color='coral', alpha=0.8, edgecolor='black')

    ax_compare.set_yticks(x)
    # Add asterisk to labels for features with negative R² (clipped to 0)
    labels = []
    for i, f in enumerate(feature_list):
        label = f.replace('_', ' ')[:20]
        if raw_linear_scores[i] < 0 or raw_nonlinear_scores[i] < 0:
            label += ' *'  # Mark clipped features
        labels.append(label)
    ax_compare.set_yticklabels(labels, fontsize=8)
    ax_compare.set_xlabel('R² Score', fontweight='bold')
    ax_compare.set_title('Predictive Power: Linear vs Non-linear\n(* = clipped from negative)',
                        fontweight='bold', fontsize=12)
    ax_compare.legend(loc='lower right')
    ax_compare.grid(alpha=0.2, axis='x')
    ax_compare.set_xlim(0, 1)

    # Panel B: Improvement from non-linearity
    ax_improvement = fig.add_subplot(gs[0, 1])

    improvement = np.array(nonlinear_scores) - np.array(linear_scores)
    sorted_idx = np.argsort(improvement)[::-1][:10]

    colors = plt.cm.RdYlGn(improvement[sorted_idx] / max(improvement[sorted_idx].max(), 0.01))
    bars = ax_improvement.barh(range(len(sorted_idx)), improvement[sorted_idx],
                               color=colors, edgecolor='black')
    ax_improvement.set_yticks(range(len(sorted_idx)))
    ax_improvement.set_yticklabels([feature_list[i].replace('_', ' ')[:20] for i in sorted_idx],
                                   fontsize=9)
    ax_improvement.set_xlabel('ΔR² (Non-linear - Linear)', fontweight='bold')
    ax_improvement.set_title('Top 10: Most Non-linear Features', fontweight='bold', fontsize=12)
    ax_improvement.invert_yaxis()
    ax_improvement.grid(alpha=0.2, axis='x')
    ax_improvement.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # Panel C: Summary statistics
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.axis('off')

    summary_text = "Predictive Performance Summary\n" + "="*50 + "\n\n"
    summary_text += f"Mean R² (Linear):      {np.mean(linear_scores):.3f}\n"
    summary_text += f"Mean R² (Non-linear):  {np.mean(nonlinear_scores):.3f}\n"
    summary_text += f"Mean Improvement:      {np.mean(improvement):.3f}\n\n"

    summary_text += "Best Predicted Features (Linear):\n"
    summary_text += "-"*50 + "\n"
    top_linear = np.argsort(linear_scores)[::-1][:5]
    for i, idx in enumerate(top_linear, 1):
        feat = feature_list[idx].replace('_', ' ').title()
        summary_text += f"{i}. {feat:<30} R²={linear_scores[idx]:.3f}\n"

    summary_text += "\nMost Non-linear Features:\n"
    summary_text += "-"*50 + "\n"
    top_nonlinear = np.argsort(improvement)[::-1][:5]
    for i, idx in enumerate(top_nonlinear, 1):
        feat = feature_list[idx].replace('_', ' ').title()
        summary_text += f"{i}. {feat:<30} Δ={improvement[idx]:+.3f}\n"

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, family='monospace',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Panel D: Feature importance matrix (which dims predict which features)
    ax_importance = fig.add_subplot(gs[1, :])

    print("Computing per-dimension importance...")

    # For top 10 features, show which dimensions are most important
    top_features = np.argsort(nonlinear_scores)[::-1][:10]
    importance_matrix = np.zeros((len(top_features), latent.shape[1]))

    for i, feat_idx in enumerate(top_features):
        y = features[feature_list[feat_idx]]
        y_train, y_test = y[:n_train], y[n_train:]

        # Train RF and get importances
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
        rf.fit(X_train, y_train)
        importance_matrix[i, :] = rf.feature_importances_

    im = ax_importance.imshow(importance_matrix, aspect='auto', cmap='YlOrRd')
    ax_importance.set_yticks(range(len(top_features)))
    ax_importance.set_yticklabels([feature_list[i].replace('_', ' ')[:20] for i in top_features],
                                  fontsize=9)
    ax_importance.set_xlabel('Latent Dimension', fontweight='bold')
    ax_importance.set_title('Feature Importance Matrix (Top 10 Predictable Features)',
                           fontweight='bold', fontsize=12)
    ax_importance.set_xticks(range(0, latent.shape[1], 5))

    cbar = plt.colorbar(im, ax=ax_importance)
    cbar.set_label('Importance', fontweight='bold')

    plt.savefig(OUTPUT_DIR / 'Figure10_Predictive_Power.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure10_Predictive_Power.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 10")
    plt.close()


def main():
    print("\n" + "="*70)
    print("DEEP LATENT SEMANTICS ANALYSIS")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}/\n")

    # Load data
    data = load_and_encode()

    # Generate figures
    figure7_glv_parameter_encoding(data)
    figure8_dimension_semantics(data)
    figure9_dynamical_regimes(data)
    figure10_predictive_power(data)

    print("\n" + "="*70)
    print("🎉 DEEP ANALYSIS COMPLETE! 🎉")
    print("="*70)
    print(f"\n✓ All figures saved to: {OUTPUT_DIR}/")
    print("\nGenerated:")
    print("  - Figure 7: GLV Parameter Encoding")
    print("  - Figure 8: Dimension Semantics")
    print("  - Figure 9: Dynamical Regimes")
    print("  - Figure 10: Predictive Power")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
