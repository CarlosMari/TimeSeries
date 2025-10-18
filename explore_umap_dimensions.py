"""
Explore different UMAP dimension combinations to find most informative views.

Based on the insight that UMAP1 often captures "sample mediocrity" rather than
interesting structure. We'll compare:
- UMAP 1 vs 2 (standard)
- UMAP 2 vs 3 (potentially more informative)
- UMAP 1 vs 3 (alternative view)

And analyze which dimensions correlate best with features.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
import umap
from tqdm import tqdm

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
})

MODEL_PATH = 'model_ckpts/model_new_arch.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Distinct cluster colors
CLUSTER_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']


def safe_overshoot(peak_vals, steady_vals, max_overshoot=10.0):
    """Safe overshoot calculation."""
    steady_vals_safe = np.maximum(steady_vals, 1e-3)
    overshoot = (peak_vals - steady_vals_safe) / steady_vals_safe
    overshoot = np.clip(overshoot, -1.0, max_overshoot)
    overshoot[steady_vals < 1e-4] = 0.0
    return overshoot


def extract_features(curves, metadata_list):
    """Extract features for analysis."""
    features = {}
    steady_vals = np.mean(curves[:, :, -10:], axis=2)
    peak_vals = np.max(curves, axis=2)

    features['steady_state_mean'] = np.mean(steady_vals, axis=1)
    features['temporal_variance'] = np.mean(np.var(curves, axis=2), axis=1)
    features['peak_time_mean'] = np.mean(np.argmax(curves, axis=2), axis=1)

    overshoot = safe_overshoot(peak_vals, steady_vals)
    features['overshoot_mean'] = np.mean(overshoot, axis=1)

    features['max_eigenvalue'] = np.array([m['max_eigenvalue'] for m in metadata_list])
    features['stability_measure'] = np.array([m['stability_measure'] for m in metadata_list])

    return features


def analyze_umap_dimensions(embedding, features):
    """
    Analyze which UMAP dimensions capture which features.

    Returns correlations for each UMAP dimension with each feature.
    """
    n_dims = embedding.shape[1]
    feature_names = list(features.keys())

    correlations = {}

    for dim in range(n_dims):
        correlations[f'UMAP{dim+1}'] = {}
        for feat_name in feature_names:
            corr, pval = spearmanr(embedding[:, dim], features[feat_name])
            correlations[f'UMAP{dim+1}'][feat_name] = {
                'correlation': corr,
                'p_value': pval
            }

    return correlations


def plot_umap_comparison(embedding, features, cluster_labels, data, output_dir):
    """
    Create comprehensive comparison of UMAP dimension combinations.
    """
    print("\n" + "="*70)
    print("UMAP DIMENSION COMPARISON")
    print("="*70)

    # Analyze correlations
    print("\nAnalyzing UMAP dimension correlations with features...")
    corr_analysis = analyze_umap_dimensions(embedding, features)

    # Print summary
    print("\nUMAP Dimension - Feature Correlations:")
    print("-" * 70)
    for umap_dim in ['UMAP1', 'UMAP2', 'UMAP3']:
        print(f"\n{umap_dim}:")
        # Sort by absolute correlation
        feat_corrs = [(feat, vals['correlation'])
                      for feat, vals in corr_analysis[umap_dim].items()]
        feat_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, corr in feat_corrs[:3]:  # Top 3
            print(f"  {feat:25s}: ρ={corr:6.3f}")

    # Create figure with 3 different UMAP views
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    n_clusters = 5

    # View 1: UMAP 1 vs 2 (STANDARD)
    print("\nGenerating UMAP 1 vs 2 view...")
    ax_row0 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Clustering view
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax_row0[0].scatter(embedding[mask, 0], embedding[mask, 1],
                          c=CLUSTER_COLORS[cluster_id], s=30, alpha=0.6,
                          edgecolors='black', linewidth=0.5,
                          label=f'C{cluster_id} (n={np.sum(mask)})')

    ax_row0[0].set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_row0[0].set_ylabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_row0[0].set_title('A. Standard View: UMAP 1 vs 2\n(Clusters)',
                         fontweight='bold', loc='left', fontsize=14)
    ax_row0[0].legend(loc='best', fontsize=9, ncol=2)
    ax_row0[0].grid(alpha=0.2, linestyle='--')

    # Temporal variance
    scatter1 = ax_row0[1].scatter(embedding[:, 0], embedding[:, 1],
                                  c=features['temporal_variance'],
                                  cmap='viridis', s=25, alpha=0.7,
                                  edgecolors='black', linewidth=0.3)
    ax_row0[1].set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_row0[1].set_ylabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_row0[1].set_title('B. UMAP 1 vs 2\n(Temporal Variance)',
                         fontweight='bold', loc='left', fontsize=14)
    plt.colorbar(scatter1, ax=ax_row0[1], label='Temporal Var.')
    ax_row0[1].grid(alpha=0.2, linestyle='--')

    # Stability
    scatter2 = ax_row0[2].scatter(embedding[:, 0], embedding[:, 1],
                                  c=features['stability_measure'],
                                  cmap='RdYlBu_r', s=25, alpha=0.7,
                                  edgecolors='black', linewidth=0.3)
    ax_row0[2].set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_row0[2].set_ylabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_row0[2].set_title('C. UMAP 1 vs 2\n(Stability)',
                         fontweight='bold', loc='left', fontsize=14)
    plt.colorbar(scatter2, ax=ax_row0[2], label='Stability')
    ax_row0[2].grid(alpha=0.2, linestyle='--')

    # View 2: UMAP 2 vs 3 (POTENTIALLY MORE INFORMATIVE!)
    print("Generating UMAP 2 vs 3 view...")
    ax_row1 = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Clustering view
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax_row1[0].scatter(embedding[mask, 1], embedding[mask, 2],
                          c=CLUSTER_COLORS[cluster_id], s=30, alpha=0.6,
                          edgecolors='black', linewidth=0.5,
                          label=f'C{cluster_id} (n={np.sum(mask)})')

    ax_row1[0].set_xlabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_row1[0].set_ylabel('UMAP 3', fontweight='bold', fontsize=13)
    ax_row1[0].set_title('D. Alternative View: UMAP 2 vs 3\n(Clusters)',
                         fontweight='bold', loc='left', fontsize=14)
    ax_row1[0].legend(loc='best', fontsize=9, ncol=2)
    ax_row1[0].grid(alpha=0.2, linestyle='--')

    # Temporal variance
    scatter3 = ax_row1[1].scatter(embedding[:, 1], embedding[:, 2],
                                  c=features['temporal_variance'],
                                  cmap='viridis', s=25, alpha=0.7,
                                  edgecolors='black', linewidth=0.3)
    ax_row1[1].set_xlabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_row1[1].set_ylabel('UMAP 3', fontweight='bold', fontsize=13)
    ax_row1[1].set_title('E. UMAP 2 vs 3\n(Temporal Variance)',
                         fontweight='bold', loc='left', fontsize=14)
    plt.colorbar(scatter3, ax=ax_row1[1], label='Temporal Var.')
    ax_row1[1].grid(alpha=0.2, linestyle='--')

    # Stability
    scatter4 = ax_row1[2].scatter(embedding[:, 1], embedding[:, 2],
                                  c=features['stability_measure'],
                                  cmap='RdYlBu_r', s=25, alpha=0.7,
                                  edgecolors='black', linewidth=0.3)
    ax_row1[2].set_xlabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_row1[2].set_ylabel('UMAP 3', fontweight='bold', fontsize=13)
    ax_row1[2].set_title('F. UMAP 2 vs 3\n(Stability)',
                         fontweight='bold', loc='left', fontsize=14)
    plt.colorbar(scatter4, ax=ax_row1[2], label='Stability')
    ax_row1[2].grid(alpha=0.2, linestyle='--')

    # View 3: UMAP 1 vs 3 (THIRD PERSPECTIVE)
    print("Generating UMAP 1 vs 3 view...")
    ax_row2 = [fig.add_subplot(gs[2, i]) for i in range(3)]

    # Clustering view
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax_row2[0].scatter(embedding[mask, 0], embedding[mask, 2],
                          c=CLUSTER_COLORS[cluster_id], s=30, alpha=0.6,
                          edgecolors='black', linewidth=0.5,
                          label=f'C{cluster_id} (n={np.sum(mask)})')

    ax_row2[0].set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_row2[0].set_ylabel('UMAP 3', fontweight='bold', fontsize=13)
    ax_row2[0].set_title('G. Third View: UMAP 1 vs 3\n(Clusters)',
                         fontweight='bold', loc='left', fontsize=14)
    ax_row2[0].legend(loc='best', fontsize=9, ncol=2)
    ax_row2[0].grid(alpha=0.2, linestyle='--')

    # Temporal variance
    scatter5 = ax_row2[1].scatter(embedding[:, 0], embedding[:, 2],
                                  c=features['temporal_variance'],
                                  cmap='viridis', s=25, alpha=0.7,
                                  edgecolors='black', linewidth=0.3)
    ax_row2[1].set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_row2[1].set_ylabel('UMAP 3', fontweight='bold', fontsize=13)
    ax_row2[1].set_title('H. UMAP 1 vs 3\n(Temporal Variance)',
                         fontweight='bold', loc='left', fontsize=14)
    plt.colorbar(scatter5, ax=ax_row2[1], label='Temporal Var.')
    ax_row2[1].grid(alpha=0.2, linestyle='--')

    # Stability
    scatter6 = ax_row2[2].scatter(embedding[:, 0], embedding[:, 2],
                                  c=features['stability_measure'],
                                  cmap='RdYlBu_r', s=25, alpha=0.7,
                                  edgecolors='black', linewidth=0.3)
    ax_row2[2].set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_row2[2].set_ylabel('UMAP 3', fontweight='bold', fontsize=13)
    ax_row2[2].set_title('I. UMAP 1 vs 3\n(Stability)',
                         fontweight='bold', loc='left', fontsize=14)
    plt.colorbar(scatter6, ax=ax_row2[2], label='Stability')
    ax_row2[2].grid(alpha=0.2, linestyle='--')

    plt.savefig(output_dir / 'UMAP_Dimension_Comparison.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'UMAP_Dimension_Comparison.pdf',
                bbox_inches='tight')
    print(f"\n✓ Saved: UMAP_Dimension_Comparison.png/pdf")
    plt.close()

    return corr_analysis


def create_improved_figure6_umap23(embedding, features, cluster_labels, data, output_dir):
    """
    Create IMPROVED Figure 6 using UMAP 2 vs 3 instead of 1 vs 2.
    """
    print("\n" + "="*70)
    print("FIGURE 6 with UMAP 2 vs 3")
    print("="*70)

    n_clusters = 5

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: UMAP 2 vs 3 clustering
    ax_a = fig.add_subplot(gs[0, :2])

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax_a.scatter(embedding[mask, 1], embedding[mask, 2],  # UMAP 2 vs 3!
                    c=CLUSTER_COLORS[cluster_id], s=30, alpha=0.6,
                    edgecolors='black', linewidth=0.5,
                    label=f'Cluster {cluster_id} (n={np.sum(mask)})')

        # Cluster centers
        center_x = np.mean(embedding[mask, 1])
        center_y = np.mean(embedding[mask, 2])
        ax_a.scatter(center_x, center_y, s=400, c=CLUSTER_COLORS[cluster_id],
                    marker='*', edgecolors='black', linewidth=3, zorder=10)
        ax_a.text(center_x, center_y, str(cluster_id),
                 ha='center', va='center', fontweight='bold',
                 fontsize=14, color='white', zorder=11)

    ax_a.set_xlabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_a.set_ylabel('UMAP 3', fontweight='bold', fontsize=13)
    ax_a.set_title('A. Dynamical Regimes (UMAP 2 vs 3, k=5)\nAlternative view excluding potentially trivial UMAP1',
                   fontweight='bold', loc='left', fontsize=14)
    ax_a.legend(loc='best', fontsize=11, framealpha=0.95, ncol=2)
    ax_a.grid(alpha=0.2, linestyle='--')

    # Panel B: Cluster characteristics (same as before)
    ax_b = fig.add_subplot(gs[0, 2])

    feature_names = ['temporal_variance', 'steady_state_mean',
                     'overshoot_mean', 'stability_measure']
    cluster_means = np.zeros((n_clusters, len(feature_names)))

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        for feat_idx, feat_name in enumerate(feature_names):
            cluster_means[cluster_id, feat_idx] = np.mean(features[feat_name][mask])

    cluster_means_norm = (cluster_means - cluster_means.min(axis=0)) / (
        cluster_means.max(axis=0) - cluster_means.min(axis=0) + 1e-8)

    im = ax_b.imshow(cluster_means_norm.T, cmap='YlOrRd', aspect='auto',
                     vmin=0, vmax=1)
    ax_b.set_xticks(range(n_clusters))
    ax_b.set_xticklabels([f'C{i}' for i in range(n_clusters)], fontsize=12)
    ax_b.set_yticks(range(len(feature_names)))
    ax_b.set_yticklabels([f.replace('_', ' ').title() for f in feature_names],
                         fontsize=11)
    ax_b.set_xlabel('Cluster', fontweight='bold', fontsize=13)
    ax_b.set_title('B. Cluster Characteristics\n(normalized feature means)',
                   fontweight='bold', loc='left', fontsize=14)

    for i in range(len(feature_names)):
        for j in range(n_clusters):
            text = ax_b.text(j, i, f'{cluster_means_norm[j, i]:.2f}',
                           ha="center", va="center",
                           color="black" if cluster_means_norm[j, i] < 0.5 else "white",
                           fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_b)
    cbar.set_label('Normalized Value', fontsize=11)

    # Panels C-E: Representative curves (same as before)
    for plot_idx in range(3):
        cluster_id = plot_idx
        ax = fig.add_subplot(gs[1, plot_idx])

        mask = cluster_labels == cluster_id
        cluster_indices = np.where(mask)[0]

        n_examples = min(15, len(cluster_indices))
        example_indices = np.random.choice(cluster_indices, n_examples, replace=False)

        for idx in example_indices:
            curve = data['curves'][idx]
            for species in range(curve.shape[0]):
                ax.plot(curve[species], alpha=0.15, color=CLUSTER_COLORS[cluster_id],
                       linewidth=1.5)

        mean_curve = np.mean(data['curves'][example_indices], axis=0)
        for species in range(mean_curve.shape[0]):
            ax.plot(mean_curve[species], color=CLUSTER_COLORS[cluster_id],
                   linewidth=3, alpha=0.9)

        ax.set_xlabel('Time', fontweight='bold', fontsize=12)
        ax.set_ylabel('Abundance', fontweight='bold', fontsize=12)
        ax.set_title(f'{chr(67+plot_idx)}. Cluster {cluster_id} Examples\n(n={np.sum(mask)})',
                    fontweight='bold', loc='left', fontsize=13)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(alpha=0.3, linestyle='--')

    plt.savefig(output_dir / 'Figure6_UMAP23_Organization.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure6_UMAP23_Organization.pdf',
                bbox_inches='tight')
    print(f"✓ Saved: Figure6_UMAP23_Organization.png/pdf")
    plt.close()


def main():
    print("="*70)
    print("EXPLORING UMAP DIMENSIONS")
    print("Testing hypothesis: UMAP1 captures 'sample mediocrity'")
    print("="*70)

    # Load data and model
    print("\nLoading data and model...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    model = LSTM_VAE(config=model_config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded {len(data['curves'])} samples")

    # Encode
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

    # Extract features
    print("\nExtracting features...")
    features = extract_features(data['curves'], data['metadata'])

    # Compute 3D UMAP
    print("\nComputing 3D UMAP...")
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_3d = reducer.fit_transform(latent_codes)
    print(f"UMAP embedding shape: {embedding_3d.shape}")

    # Clustering (on original 2D for consistency)
    print("\nClustering...")
    embedding_2d = embedding_3d[:, :2]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(latent_codes)

    # Analyze correlations
    corr_analysis = plot_umap_comparison(embedding_3d, features, cluster_labels,
                                         data, OUTPUT_DIR)

    # Create improved Figure 6 with UMAP 2 vs 3
    create_improved_figure6_umap23(embedding_3d, features, cluster_labels,
                                   data, OUTPUT_DIR)

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. UMAP_Dimension_Comparison.png/pdf")
    print("     - Shows all 3 UMAP dimension combinations")
    print("     - Compare which view is most informative")
    print("\n  2. Figure6_UMAP23_Organization.png/pdf")
    print("     - Figure 6 using UMAP 2 vs 3 instead of 1 vs 2")
    print("     - May reveal more interesting structure")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print("\nCompare the correlation strengths:")

    for dim_pair, desc in [('UMAP1', 'Standard first component'),
                           ('UMAP2', 'Second component'),
                           ('UMAP3', 'Third component')]:
        print(f"\n{dim_pair} ({desc}):")
        feat_corrs = [(feat, vals['correlation'])
                      for feat, vals in corr_analysis[dim_pair].items()]
        feat_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_feat, top_corr = feat_corrs[0]
        print(f"  Strongest: {top_feat} (ρ={top_corr:.3f})")

    print("\n→ If UMAP2 or UMAP3 show stronger feature correlations,")
    print("  use Figure6_UMAP23_Organization.png for your paper!")
    print("="*70)


if __name__ == "__main__":
    main()
