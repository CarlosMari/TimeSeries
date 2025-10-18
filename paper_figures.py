"""
IMPROVED Publication Figures - Enhanced Clarity Version

Improvements:
- Figure 1: Clearer PCA plot, better UMAP contrast, improved labels
- Figure 4: Simplified species display, clearer reconstruction comparison
- Figure 5: Fixed axis scales, better distribution comparison
- Figure 6: Distinct cluster colors, clearer representative curves
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import umap
from tqdm import tqdm
import warnings

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE

# Enhanced publication settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
})

# Configuration
MODEL_PATH = 'model_ckpts/model_new_arch.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Enhanced color palettes
COLORS_DISTINCT = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                   '#ffff33', '#a65628', '#f781bf', '#999999']  # ColorBrewer Set1


def safe_overshoot(peak_vals, steady_vals, max_overshoot=10.0):
    """Safe overshoot calculation."""
    steady_vals_safe = np.maximum(steady_vals, 1e-3)
    overshoot = (peak_vals - steady_vals_safe) / steady_vals_safe
    overshoot = np.clip(overshoot, -1.0, max_overshoot)
    overshoot[steady_vals < 1e-4] = 0.0
    return overshoot


def extract_all_features(curves, metadata_list):
    """Extract comprehensive feature set."""
    features = {}
    steady_vals = np.mean(curves[:, :, -10:], axis=2)
    peak_vals = np.max(curves, axis=2)

    features['steady_state_mean'] = np.mean(steady_vals, axis=1)
    features['steady_state_std'] = np.std(steady_vals, axis=1)
    features['peak_time_mean'] = np.mean(np.argmax(curves, axis=2), axis=1)
    features['temporal_variance'] = np.mean(np.var(curves, axis=2), axis=1)

    overshoot = safe_overshoot(peak_vals, steady_vals)
    features['overshoot_mean'] = np.mean(overshoot, axis=1)

    initial_growth = np.gradient(curves[:, :, :10], axis=2).mean(axis=2)
    features['initial_growth_mean'] = np.mean(initial_growth, axis=1)

    features['max_eigenvalue'] = np.array([m['max_eigenvalue'] for m in metadata_list])
    features['diagonal_strength'] = np.array([m['diagonal_strength'] for m in metadata_list])
    features['stability_measure'] = np.array([m['stability_measure'] for m in metadata_list])

    return features


# ============================================================================
# FIGURE 1: IMPROVED Latent Space Structure
# ============================================================================

def figure1_improved(latent_codes, features, output_dir):
    """
    IMPROVED Figure 1: Better clarity, colors, and labels.
    """
    print("\n" + "="*70)
    print("FIGURE 1 (IMPROVED): Latent Space Structure")
    print("="*70)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: IMPROVED PCA with clearer styling
    ax_a = fig.add_subplot(gs[0, 0])
    print("  Panel A: Computing PCA...")

    pca = PCA(n_components=min(50, latent_codes.shape[1]))
    pca.fit(latent_codes)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Twin axes for better clarity
    ax_a_twin = ax_a.twinx()

    # Bar plot with gradient colors
    n_components = len(explained_var)
    colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.9, n_components))
    bars = ax_a.bar(range(1, n_components+1), explained_var,
                     alpha=0.8, color=colors_gradient, edgecolor='navy', linewidth=0.5)

    ax_a.set_xlabel('Principal Component', fontweight='bold', fontsize=13)
    ax_a.set_ylabel('Explained Variance Ratio', fontweight='bold',
                    color='navy', fontsize=13)
    ax_a.tick_params(axis='y', labelcolor='navy', labelsize=11)

    # Cumulative line with markers
    line = ax_a_twin.plot(range(1, n_components+1), cumulative_var,
                          'ro-', linewidth=3, markersize=6,
                          markerfacecolor='red', markeredgecolor='darkred',
                          label='Cumulative')
    ax_a_twin.set_ylabel('Cumulative Variance', fontweight='bold',
                         color='darkred', fontsize=13)
    ax_a_twin.tick_params(axis='y', labelcolor='darkred', labelsize=11)

    # Reference lines with labels
    ax_a_twin.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax_a_twin.axhline(y=0.95, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    ax_a_twin.text(n_components*0.7, 0.91, '90%', fontsize=11,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_a_twin.text(n_components*0.7, 0.96, '95%', fontsize=11,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    n_90 = np.argmax(cumulative_var >= 0.9) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1

    ax_a.set_title(f'A. Effective Dimensionality\n90% variance: {n_90} dims | 95% variance: {n_95} dims',
                   fontweight='bold', loc='left', fontsize=14)
    ax_a.set_xlim(0, min(51, n_components+1))
    ax_a.grid(alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax_a.set_ylim(0, max(explained_var) * 1.1)
    ax_a_twin.set_ylim(0, 1.05)

    # Panel B & C: IMPROVED UMAP with better colormaps
    print("  Panel B-C: Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_codes)

    # Panel B: Temporal variance with viridis
    ax_b = fig.add_subplot(gs[0, 1])
    tv = features['temporal_variance']
    scatter_b = ax_b.scatter(embedding[:, 0], embedding[:, 1],
                            c=tv, cmap='viridis', s=25, alpha=0.7,
                            edgecolors='black', linewidth=0.3)
    ax_b.set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_b.set_ylabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_b.set_title('B. UMAP: Temporal Variance\n(Dynamical Feature)',
                   fontweight='bold', loc='left', fontsize=14)
    cbar_b = plt.colorbar(scatter_b, ax=ax_b, fraction=0.046)
    cbar_b.set_label('Temporal Variance', fontsize=12)
    ax_b.grid(alpha=0.2, linestyle='--')

    # Panel C: Stability with RdYlBu (diverging)
    ax_c = fig.add_subplot(gs[1, 0])
    stability = features['stability_measure']
    scatter_c = ax_c.scatter(embedding[:, 0], embedding[:, 1],
                            c=stability, cmap='RdYlBu_r', s=25, alpha=0.7,
                            edgecolors='black', linewidth=0.3)
    ax_c.set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_c.set_ylabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_c.set_title('C. UMAP: Stability Measure\n(Interaction Property)',
                   fontweight='bold', loc='left', fontsize=14)
    cbar_c = plt.colorbar(scatter_c, ax=ax_c, fraction=0.046)
    cbar_c.set_label('Stability', fontsize=12)
    ax_c.grid(alpha=0.2, linestyle='--')

    # Panel D: IMPROVED variance distribution
    ax_d = fig.add_subplot(gs[1, 1])
    latent_vars = np.var(latent_codes, axis=0)
    sorted_vars = np.sort(latent_vars)[::-1]

    # Use gradient colors
    colors_var = plt.cm.Oranges(np.linspace(0.4, 0.9, len(sorted_vars)))
    ax_d.bar(range(len(sorted_vars)), sorted_vars, color=colors_var,
             alpha=0.9, edgecolor='darkorange', linewidth=0.8)
    ax_d.set_xlabel('Latent Dimension (sorted by variance)',
                    fontweight='bold', fontsize=13)
    ax_d.set_ylabel('Variance', fontweight='bold', fontsize=13)
    ax_d.set_title('D. Latent Dimension Variance Distribution',
                   fontweight='bold', loc='left', fontsize=14)
    ax_d.grid(alpha=0.3, axis='y', linestyle='-', linewidth=0.5)

    # Add statistics box
    mean_var = np.mean(latent_vars)
    std_var = np.std(latent_vars)
    max_var = np.max(latent_vars)
    ax_d.axhline(y=mean_var, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_var:.3f}')

    stats_text = f'Mean: {mean_var:.3f}\nStd: {std_var:.3f}\nMax: {max_var:.3f}'
    ax_d.text(0.98, 0.97, stats_text, transform=ax_d.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             fontsize=11)

    plt.savefig(output_dir / 'Figure1_Latent_Structure_IMPROVED.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure1_Latent_Structure_IMPROVED.pdf',
                bbox_inches='tight')
    print(f"  ✓ Saved: Figure1_Latent_Structure_IMPROVED")
    plt.close()

    return embedding, pca


# ============================================================================
# FIGURE 4: IMPROVED Generative Capabilities
# ============================================================================

def figure4_improved(model, data, latent_codes, output_dir):
    """
    IMPROVED Figure 4: Clearer reconstructions, better layout.
    """
    print("\n" + "="*70)
    print("FIGURE 4 (IMPROVED): Generative Capabilities")
    print("="*70)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    curves = data['curves']

    # Panel A-C: IMPROVED Reconstructions - show only 3 species for clarity
    print("  Computing reconstructions...")
    indices = [100, 300, 500]
    species_to_show = [0, 2, 4]  # Show species 1, 3, 5

    with torch.no_grad():
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(gs[0, i])

            original = curves[idx]
            latent = latent_codes[idx:idx+1]

            latent_tensor = torch.FloatTensor(latent).to(DEVICE)
            reconstructed, _ = model.decode(latent_tensor)
            recon = reconstructed[0].cpu().numpy()

            seq_len = min(original.shape[1], recon.shape[1])
            original_trimmed = original[:, :seq_len]
            recon_trimmed = recon[:, :seq_len]

            # Plot only selected species with distinct colors
            for plot_idx, species_idx in enumerate(species_to_show):
                color = COLORS_DISTINCT[plot_idx]
                ax.plot(original_trimmed[species_idx], '-', alpha=0.8,
                       linewidth=2.5, color=color,
                       label=f'Sp{species_idx+1} Orig' if i == 0 else '')
                ax.plot(recon_trimmed[species_idx], '--', alpha=0.9,
                       linewidth=2.5, color=color,
                       label=f'Sp{species_idx+1} Recon' if i == 0 else '')

            mse = np.mean((original_trimmed - recon_trimmed)**2)
            ax.set_title(f'{chr(65+i)}. Reconstruction Example {i+1}\nMSE = {mse:.4f}',
                        fontweight='bold', loc='left', fontsize=13)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Abundance', fontsize=12)
            ax.set_ylim(-0.05, 1.15)
            ax.grid(alpha=0.3, linestyle='--')

            if i == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                         ncol=1, fontsize=10, framealpha=0.95)

    # Panels D-H: IMPROVED Interpolations - one species, clearer
    print("  Generating interpolations...")
    start_idx, end_idx = 50, 250
    n_steps = 5

    with torch.no_grad():
        for step in range(n_steps):
            if step < 3:
                ax = fig.add_subplot(gs[1, step])
            else:
                ax = fig.add_subplot(gs[2, step - 3])

            alpha = step / (n_steps - 1)
            interpolated_latent = (1 - alpha) * latent_codes[start_idx] + alpha * latent_codes[end_idx]

            latent_tensor = torch.FloatTensor(interpolated_latent[np.newaxis, :]).to(DEVICE)
            decoded, _ = model.decode(latent_tensor)
            curve = decoded[0].cpu().numpy()

            # Show only 3 species
            for plot_idx, species_idx in enumerate(species_to_show):
                color = COLORS_DISTINCT[plot_idx]
                ax.plot(curve[species_idx], alpha=0.9, linewidth=2.5,
                       color=color, label=f'Species {species_idx+1}')

            label = 'Start' if step == 0 else ('End' if step == n_steps-1 else f'Interp. α={alpha:.2f}')
            panel_label = chr(68 + step)  # D, E, F, G, H
            ax.set_title(f'{panel_label}. {label}',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel('Abundance', fontsize=11)
            ax.set_ylim(-0.05, 1.15)
            ax.grid(alpha=0.3, linestyle='--')

            if step == 0:
                ax.legend(fontsize=9, loc='upper right')

    # Panels I-K: IMPROVED Random sampling
    print("  Generating random samples...")
    with torch.no_grad():
        for i in range(3):
            ax = fig.add_subplot(gs[3 if n_steps >= 3 else 2, i])

            random_latent = np.random.randn(1, latent_codes.shape[1])
            latent_tensor = torch.FloatTensor(random_latent).to(DEVICE)
            decoded, _ = model.decode(latent_tensor)
            curve = decoded[0].cpu().numpy()

            for plot_idx, species_idx in enumerate(species_to_show):
                color = COLORS_DISTINCT[plot_idx]
                ax.plot(curve[species_idx], alpha=0.9, linewidth=2.5, color=color)

            panel_label = chr(73 + i)  # I, J, K
            ax.set_title(f'{panel_label}. Random Sample {i+1}\nfrom N(0,I)',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel('Abundance', fontsize=11)
            ax.set_ylim(-0.05, 1.15)
            ax.grid(alpha=0.3, linestyle='--')

    plt.savefig(output_dir / 'Figure4_Generative_Quality_IMPROVED.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure4_Generative_Quality_IMPROVED.pdf',
                bbox_inches='tight')
    print(f"  ✓ Saved: Figure4_Generative_Quality_IMPROVED")
    plt.close()


# ============================================================================
# FIGURE 5: IMPROVED GLV Validation - FIXED SCALES
# ============================================================================

def figure5_improved(model, data, latent_codes, output_dir):
    """
    IMPROVED Figure 5: Fixed axis scales, better comparison.
    """
    print("\n" + "="*70)
    print("FIGURE 5 (IMPROVED): GLV Validation")
    print("="*70)

    from sklearn.linear_model import LinearRegression

    curves = data['curves']
    n_samples = min(500, len(curves))

    print(f"  Validating {n_samples} samples...")

    r2_original = []
    r2_reconstructed = []

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Computing GLV fits"):
            original = curves[idx]

            latent = latent_codes[idx:idx+1]
            latent_tensor = torch.FloatTensor(latent).to(DEVICE)
            reconstructed, _ = model.decode(latent_tensor)
            recon = reconstructed[0].cpu().numpy()

            seq_len = min(original.shape[1], recon.shape[1])
            original_trimmed = original[:, :seq_len]
            recon_trimmed = recon[:, :seq_len]

            for curve in [original_trimmed, recon_trimmed]:
                t = np.arange(curve.shape[1])
                r2_species = []

                for species in range(curve.shape[0]):
                    x_i = np.maximum(curve[species], 1e-9)
                    log_x = np.log(x_i)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        d_log_x = np.gradient(log_x, t)

                    X = curve.T
                    y = d_log_x

                    if not np.any(np.isnan(X)) and not np.any(np.isnan(y)):
                        model_lr = LinearRegression()
                        model_lr.fit(X, y)
                        r2 = model_lr.score(X, y)
                        r2_species.append(r2)

                if curve is original_trimmed:
                    r2_original.append(np.mean(r2_species) if r2_species else 0)
                else:
                    r2_reconstructed.append(np.mean(r2_species) if r2_species else 0)

    r2_original = np.array(r2_original)
    r2_reconstructed = np.array(r2_reconstructed)

    # Create figure with FIXED SCALES
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: IMPROVED R² distributions with same x-axis
    ax_a = fig.add_subplot(gs[0, 0])

    # Use same bins and range
    bins = np.linspace(-0.5, 1.0, 40)

    ax_a.hist(r2_original, bins=bins, alpha=0.6, label='Original',
             color='#377eb8', density=True, edgecolor='navy', linewidth=1.2)
    ax_a.hist(r2_reconstructed, bins=bins, alpha=0.6, label='Reconstructed',
             color='#e41a1c', density=True, edgecolor='darkred', linewidth=1.2)

    mean_orig = np.mean(r2_original)
    mean_recon = np.mean(r2_reconstructed)

    ax_a.axvline(mean_orig, color='#377eb8', linestyle='--', linewidth=2.5,
                label=f'Mean Orig: {mean_orig:.3f}')
    ax_a.axvline(mean_recon, color='#e41a1c', linestyle='--', linewidth=2.5,
                label=f'Mean Recon: {mean_recon:.3f}')

    ax_a.set_xlabel('R² Score (GLV Fit)', fontweight='bold', fontsize=13)
    ax_a.set_ylabel('Density', fontweight='bold', fontsize=13)
    ax_a.set_title('A. GLV Dynamics Fit Quality', fontweight='bold',
                   loc='left', fontsize=14)
    ax_a.legend(fontsize=11, framealpha=0.95)
    ax_a.grid(alpha=0.3, linestyle='--')
    ax_a.set_xlim(-0.5, 1.0)  # FIXED SCALE

    # Panel B: IMPROVED Feature preservation with fixed aspect
    ax_b = fig.add_subplot(gs[0, 1])

    subset_curves_orig = curves[:n_samples]

    with torch.no_grad():
        latent_subset = latent_codes[:n_samples]
        latent_tensor = torch.FloatTensor(latent_subset).to(DEVICE)

        reconstructed_all = []
        batch_size = 64
        for i in range(0, len(latent_tensor), batch_size):
            batch = latent_tensor[i:i+batch_size]
            recon, _ = model.decode(batch)
            reconstructed_all.append(recon.cpu().numpy())

        subset_curves_recon = np.concatenate(reconstructed_all, axis=0)

    seq_len = min(subset_curves_orig.shape[2], subset_curves_recon.shape[2])
    subset_curves_orig_trimmed = subset_curves_orig[:, :, :seq_len]
    subset_curves_recon_trimmed = subset_curves_recon[:, :, :seq_len]

    tv_orig = np.mean(np.var(subset_curves_orig_trimmed, axis=2), axis=1)
    tv_recon = np.mean(np.var(subset_curves_recon_trimmed, axis=2), axis=1)

    # Hexbin for density
    hexbin = ax_b.hexbin(tv_orig, tv_recon, gridsize=30, cmap='Blues',
                         mincnt=1, alpha=0.7, edgecolors='black', linewidths=0.3)

    # Identity line
    max_val = max(tv_orig.max(), tv_recon.max())
    ax_b.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, linewidth=2.5,
             label='Perfect preservation')

    corr, _ = pearsonr(tv_orig, tv_recon)

    ax_b.set_xlabel('Original Temporal Variance', fontweight='bold', fontsize=13)
    ax_b.set_ylabel('Reconstructed Temporal Variance', fontweight='bold', fontsize=13)
    ax_b.set_title(f'B. Feature Preservation\n(Pearson ρ={corr:.3f})',
                   fontweight='bold', loc='left', fontsize=14)
    ax_b.legend(fontsize=11)
    ax_b.grid(alpha=0.3, linestyle='--')

    # Equal aspect ratio
    ax_b.set_aspect('equal', adjustable='box')
    ax_b.set_xlim(0, max_val * 1.05)
    ax_b.set_ylim(0, max_val * 1.05)

    plt.colorbar(hexbin, ax=ax_b, label='Count')

    # Panel C: IMPROVED Diversity with better alignment
    ax_c = fig.add_subplot(gs[0, 2])

    threshold = 0.05
    n_surv_orig = np.sum(subset_curves_orig_trimmed[:, :, -1] > threshold, axis=1)
    n_surv_recon = np.sum(subset_curves_recon_trimmed[:, :, -1] > threshold, axis=1)

    bins = np.arange(0, 8) - 0.4
    width = 0.35

    # Side-by-side bars for better comparison
    x = np.arange(8)
    hist_orig, _ = np.histogram(n_surv_orig, bins=np.arange(0, 9))
    hist_recon, _ = np.histogram(n_surv_recon, bins=np.arange(0, 9))

    ax_c.bar(x - width/2, hist_orig, width, alpha=0.7, label='Original',
            color='#377eb8', edgecolor='navy', linewidth=1.2)
    ax_c.bar(x + width/2, hist_recon, width, alpha=0.7, label='Reconstructed',
            color='#e41a1c', edgecolor='darkred', linewidth=1.2)

    ax_c.set_xlabel('Number of Surviving Species', fontweight='bold', fontsize=13)
    ax_c.set_ylabel('Count', fontweight='bold', fontsize=13)
    ax_c.set_title(f'C. Diversity Preservation\n(Mean: Orig={np.mean(n_surv_orig):.2f}, Recon={np.mean(n_surv_recon):.2f})',
                   fontweight='bold', loc='left', fontsize=14)
    ax_c.legend(fontsize=11, framealpha=0.95)
    ax_c.grid(alpha=0.3, axis='y', linestyle='--')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(x)
    ax_c.set_xlim(-0.5, 7.5)

    plt.savefig(output_dir / 'Figure5_GLV_Validation_IMPROVED.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure5_GLV_Validation_IMPROVED.pdf',
                bbox_inches='tight')
    print(f"  ✓ Saved: Figure5_GLV_Validation_IMPROVED")
    plt.close()


# ============================================================================
# FIGURE 6: IMPROVED Latent Organization - DISTINCT COLORS
# ============================================================================

def figure6_improved(latent_codes, features, embedding, output_dir, data):
    """
    IMPROVED Figure 6: Distinct cluster colors, clearer curves.
    """
    print("\n" + "="*70)
    print("FIGURE 6 (IMPROVED): Latent Organization")
    print("="*70)

    from sklearn.cluster import KMeans

    print("  Performing k-means clustering...")
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(latent_codes)

    # DISTINCT cluster colors
    cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: IMPROVED UMAP with distinct colors
    ax_a = fig.add_subplot(gs[0, :2])

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax_a.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=cluster_colors[cluster_id], s=30, alpha=0.6,
                    edgecolors='black', linewidth=0.5,
                    label=f'Cluster {cluster_id} (n={np.sum(mask)})')

    # Plot cluster centers
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        center_x = np.mean(embedding[mask, 0])
        center_y = np.mean(embedding[mask, 1])
        ax_a.scatter(center_x, center_y, s=400, c=cluster_colors[cluster_id],
                    marker='*', edgecolors='black', linewidth=3, zorder=10)
        ax_a.text(center_x, center_y, str(cluster_id),
                 ha='center', va='center', fontweight='bold',
                 fontsize=14, color='white', zorder=11)

    ax_a.set_xlabel('UMAP 1', fontweight='bold', fontsize=13)
    ax_a.set_ylabel('UMAP 2', fontweight='bold', fontsize=13)
    ax_a.set_title('A. Dynamical Regimes (k-means clustering, k=5)',
                   fontweight='bold', loc='left', fontsize=14)
    ax_a.legend(loc='best', fontsize=11, framealpha=0.95, ncol=2)
    ax_a.grid(alpha=0.2, linestyle='--')

    # Panel B: IMPROVED cluster characteristics
    ax_b = fig.add_subplot(gs[0, 2])

    feature_names = ['temporal_variance', 'steady_state_mean',
                     'overshoot_mean', 'stability_measure']
    cluster_means = np.zeros((n_clusters, len(feature_names)))

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        for feat_idx, feat_name in enumerate(feature_names):
            cluster_means[cluster_id, feat_idx] = np.mean(features[feat_name][mask])

    # Normalize
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

    # Add values as text
    for i in range(len(feature_names)):
        for j in range(n_clusters):
            text = ax_b.text(j, i, f'{cluster_means_norm[j, i]:.2f}',
                           ha="center", va="center", color="black" if cluster_means_norm[j, i] < 0.5 else "white",
                           fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_b)
    cbar.set_label('Normalized Value', fontsize=11)

    # Panels C-E: IMPROVED representative curves with distinct colors
    print("  Selecting representative curves...")

    for plot_idx in range(3):
        cluster_id = plot_idx
        ax = fig.add_subplot(gs[1, plot_idx])

        mask = cluster_labels == cluster_id
        cluster_indices = np.where(mask)[0]

        n_examples = min(15, len(cluster_indices))
        example_indices = np.random.choice(cluster_indices, n_examples, replace=False)

        # Plot all 7 species but with alpha
        for idx in example_indices:
            curve = data['curves'][idx]
            for species in range(curve.shape[0]):
                ax.plot(curve[species], alpha=0.15, color=cluster_colors[cluster_id],
                       linewidth=1.5)

        # Plot mean curve with strong color
        mean_curve = np.mean(data['curves'][example_indices], axis=0)
        for species in range(mean_curve.shape[0]):
            ax.plot(mean_curve[species], color=cluster_colors[cluster_id],
                   linewidth=3, alpha=0.9, label=f'Sp{species+1} mean' if plot_idx == 0 and species < 3 else '')

        ax.set_xlabel('Time', fontweight='bold', fontsize=12)
        ax.set_ylabel('Abundance', fontweight='bold', fontsize=12)
        ax.set_title(f'{chr(67+plot_idx)}. Cluster {cluster_id} Examples\n(n={np.sum(mask)})',
                    fontweight='bold', loc='left', fontsize=13)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(alpha=0.3, linestyle='--')

        if plot_idx == 0:
            ax.legend(fontsize=9, loc='upper right', ncol=3)

    plt.savefig(output_dir / 'Figure6_Latent_Organization_IMPROVED.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'Figure6_Latent_Organization_IMPROVED.pdf',
                bbox_inches='tight')
    print(f"  ✓ Saved: Figure6_Latent_Organization_IMPROVED")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("IMPROVED FIGURES GENERATION")
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
    features = extract_all_features(data['curves'], data['metadata'])

    # Generate improved figures
    embedding, pca = figure1_improved(latent_codes, features, OUTPUT_DIR)

    figure4_improved(model, data, latent_codes, OUTPUT_DIR)

    figure5_improved(model, data, latent_codes, OUTPUT_DIR)

    figure6_improved(latent_codes, features, embedding, OUTPUT_DIR, data)

    print("\n" + "="*70)
    print("IMPROVED FIGURES COMPLETE!")
    print("="*70)
    print(f"\nGenerated in: {OUTPUT_DIR}/")
    print("  - Figure1_Latent_Structure_IMPROVED.png/pdf")
    print("  - Figure4_Generative_Quality_IMPROVED.png/pdf")
    print("  - Figure5_GLV_Validation_IMPROVED.png/pdf")
    print("  - Figure6_Latent_Organization_IMPROVED.png/pdf")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
