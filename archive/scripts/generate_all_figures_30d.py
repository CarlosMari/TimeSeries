"""
Master script to regenerate ALL paper figures with the 30D latent model.

This script generates all figures with "_30" suffix to avoid overwriting 50D results.
Allows direct comparison between 30D and 50D models.
"""

import os
import sys

# Update model configuration globally
MODEL_CONFIG_30D = {
    "in_channels": 7,
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,  # CHANGED from 50
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

MODEL_PATH_30D = 'model_ckpts/model_final_30_conditioned.pth'
OUTPUT_SUFFIX = '_30'  # Add to all output filenames

print("="*80)
print("REGENERATING ALL FIGURES WITH 30D LATENT MODEL")
print("="*80)
print(f"Model: {MODEL_PATH_30D}")
print(f"Latent dimension: {MODEL_CONFIG_30D['latent_dim']}")
print(f"Output suffix: {OUTPUT_SUFFIX}")
print("="*80)
print()

# Import after setting configuration
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from scipy.fft import fft, fftfreq
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.0,
})

from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE

# Paths
TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
OUTPUT_DIR = 'final figures/'

SPECIES_NAMES = ['Species 1', 'Species 2', 'Species 3', 'Species 4',
                 'Species 5', 'Species 6', 'Species 7']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2']


def load_model_and_data():
    """Load the 30D model and test data."""
    print("Loading 30D model and data...")

    model = LSTM_VAE(config=MODEL_CONFIG_30D)
    model.load_state_dict(torch.load(MODEL_PATH_30D, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)

    test_data = data_package['data']
    test_max_vals = data_package['reconstruction_max_values']

    print(f"Model loaded. Test samples: {len(test_data)}")
    return model, test_data, test_max_vals


def save_figure(fig, name):
    """Save figure with _30 suffix."""
    base_name = name.replace('.png', '').replace('.pdf', '')
    output_path = f'{OUTPUT_DIR}{base_name}{OUTPUT_SUFFIX}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {base_name}{OUTPUT_SUFFIX}.png/pdf")
    plt.close()


print("\n" + "="*80)
print("STARTING FIGURE GENERATION SEQUENCE")
print("="*80)

# Load model and data once
model, test_data, test_max_vals = load_model_and_data()

# ============================================================================
# FIGURE SET 1: Reconstruction Quality
# ============================================================================
print("\n[1/6] Generating Reconstruction Quality Figures...")

from generate_reconstruction_quality_figures import (
    compute_reconstruction_metrics,
    generate_reconstruction_examples_figure,
    generate_reconstruction_metrics_figure,
    generate_error_analysis_figure,
    generate_max_value_prediction_figure
)

# Update the module's configuration
import generate_reconstruction_quality_figures as recon_module
recon_module.model_config = MODEL_CONFIG_30D
recon_module.MODEL_PATH = MODEL_PATH_30D

# Compute metrics
all_metrics = compute_reconstruction_metrics(model, test_data, test_max_vals, n_samples=1000)

# Generate figures
fig1 = generate_reconstruction_examples_figure(model, test_data, test_max_vals)
save_figure(fig1, 'fig_reconstruction_examples')

fig2 = generate_reconstruction_metrics_figure(all_metrics)
save_figure(fig2, 'fig_reconstruction_metrics')

fig3 = generate_error_analysis_figure(all_metrics)
save_figure(fig3, 'fig_reconstruction_error_analysis')

fig4 = generate_max_value_prediction_figure(all_metrics)
save_figure(fig4, 'fig_max_value_prediction')

print("  Reconstruction quality figures complete!")

# ============================================================================
# FIGURE SET 2: Variance Explained (PCA)
# ============================================================================
print("\n[2/6] Generating Variance Explained Figure...")

# Encode to latent space
print("  Encoding test samples to latent space...")
n_samples = min(5000, len(test_data))
latent_codes = np.zeros((n_samples, MODEL_CONFIG_30D['latent_dim']))

model.eval()
with torch.no_grad():
    for i in tqdm(range(n_samples), desc="  Encoding"):
        x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        x_rnn = x_norm.permute(0, 2, 1)
        _, (h_n, _) = model.encoder_rnn(x_rnn)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

        z_shape = model.shape_projector(encoded_summary)
        z_scale = model.scale_encoder(max_vals)
        bottleneck_input = torch.cat([z_shape, z_scale], dim=1)

        mu = model.fc_mu(bottleneck_input)
        latent_codes[i] = mu.cpu().numpy()[0]

# PCA
print("  Computing PCA...")
pca = PCA()
pca.fit(latent_codes)
variance_ratio = pca.explained_variance_ratio_
cumsum = np.cumsum(variance_ratio)

n_90 = np.argmax(cumsum >= 0.90) + 1
n_95 = np.argmax(cumsum >= 0.95) + 1
n_99 = np.argmax(cumsum >= 0.99) + 1

print(f"  Components for 90% variance: {n_90}/{len(variance_ratio)}")
print(f"  Components for 95% variance: {n_95}/{len(variance_ratio)}")
print(f"  Components for 99% variance: {n_99}/{len(variance_ratio)}")

# Create figure
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 1, figure=fig, hspace=0.3)

# Panel 1: Full spectrum
ax1 = fig.add_subplot(gs[0])
n_dims = len(variance_ratio)
x_pos = np.arange(n_dims)

bars = ax1.bar(x_pos, variance_ratio * 100, alpha=0.7, color='steelblue',
               edgecolor='black', linewidth=0.8, label='Individual PC')

ax1_twin = ax1.twinx()
line = ax1_twin.plot(x_pos, cumsum * 100, 'ro-', linewidth=2.5,
                     markersize=4, label='Cumulative', alpha=0.8)

ax1_twin.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='90%')
ax1_twin.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='95%')
ax1_twin.axhline(y=99, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='99%')

ax1.axvline(x=n_90-1, color='green', linestyle=':', linewidth=1.5, alpha=0.4)
ax1.axvline(x=n_95-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.4)
ax1.axvline(x=n_99-1, color='red', linestyle=':', linewidth=1.5, alpha=0.4)

ax1.set_xlabel('Principal Component Index', fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
ax1_twin.set_ylabel('Cumulative Variance Explained (%)', fontweight='bold', color='red')
ax1.set_title(f'Latent Space Structure: PCA Variance Explained (30D Latent Space)',
              fontweight='bold', fontsize=13)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='right', framealpha=0.95)

ax1.grid(alpha=0.3, axis='y')
ax1.set_xlim([-0.5, n_dims - 0.5])
ax1.set_ylim([0, max(variance_ratio * 100) * 1.1])
ax1_twin.set_ylim([0, 105])
ax1_twin.tick_params(axis='y', labelcolor='red')

ax1.text(n_90-1, ax1.get_ylim()[1] * 0.95, f'PC{n_90}\n90%',
         ha='center', va='top', fontsize=9, color='green', fontweight='bold')

# Panel 2: Detailed view
ax2 = fig.add_subplot(gs[1])
n_show = min(20, n_dims)
x_pos_zoom = np.arange(n_show)

bars2 = ax2.bar(x_pos_zoom, variance_ratio[:n_show] * 100,
                alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)

ax2_twin = ax2.twinx()
ax2_twin.plot(x_pos_zoom, cumsum[:n_show] * 100, 'ro-',
              linewidth=2.5, markersize=6, alpha=0.8)

for i in range(min(5, n_show)):
    bars2[i].set_color('darkblue')
    bars2[i].set_alpha(0.9)

ax2.set_xlabel('Principal Component Index', fontweight='bold')
ax2.set_ylabel('Variance Explained (%)', fontweight='bold')
ax2_twin.set_ylabel('Cumulative Variance Explained (%)', fontweight='bold', color='red')
ax2.set_title(f'First {n_show} Principal Components (Detail View)',
              fontweight='bold', fontsize=13)

for i, (bar, val) in enumerate(zip(bars2, variance_ratio[:n_show] * 100)):
    if i < 10:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

ax2.grid(alpha=0.3, axis='y')
ax2.set_xlim([-0.5, n_show - 0.5])
ax2_twin.set_ylim([0, 105])
ax2_twin.tick_params(axis='y', labelcolor='red')

plt.suptitle('Variance Explained: 30D Latent Model', fontweight='bold', fontsize=16)

save_figure(fig, 'fig_variance_explained')
print("  Variance explained figure complete!")

# ============================================================================
# FIGURE SET 3: Latent Collapse Investigation
# ============================================================================
print("\n[3/6] Generating Latent Collapse Analysis...")

# Analyze latent space
print("  Analyzing latent space for collapse...")
n_analyze = min(5000, len(test_data))
mus = []
log_vars = []
zs = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(n_analyze), desc="  Encoding for collapse analysis"):
        x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        x_rnn = x_norm.permute(0, 2, 1)
        _, (h_n, _) = model.encoder_rnn(x_rnn)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

        z_shape = model.shape_projector(encoded_summary)
        z_scale = model.scale_encoder(max_vals)
        bottleneck_input = torch.cat([z_shape, z_scale], dim=1)

        mu = model.fc_mu(bottleneck_input)
        log_var = model.fc_log_var(bottleneck_input)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        mus.append(mu.cpu().numpy()[0])
        log_vars.append(log_var.cpu().numpy()[0])
        zs.append(z.cpu().numpy()[0])

mus = np.array(mus)
log_vars = np.array(log_vars)
zs = np.array(zs)

mu_vars = np.var(mus, axis=0)
z_vars = np.var(zs, axis=0)
mean_log_vars = np.mean(log_vars, axis=0)
mean_vars = np.exp(mean_log_vars)

threshold = 0.01
collapsed_dims = np.where(mu_vars < threshold)[0]
active_dims = np.where(mu_vars >= threshold)[0]

print(f"  Collapsed dimensions: {len(collapsed_dims)}/{len(mu_vars)}")
print(f"  Active dimensions: {len(active_dims)}/{len(mu_vars)}")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.bar(range(len(mu_vars)), mu_vars, color='steelblue', alpha=0.7)
ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
ax.set_xlabel('Latent Dimension', fontweight='bold')
ax.set_ylabel('Variance of μ', fontweight='bold')
ax.set_title('Variance of Posterior Mean (μ) - 30D Model', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.bar(range(len(z_vars)), z_vars, color='orange', alpha=0.7)
ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
ax.set_xlabel('Latent Dimension', fontweight='bold')
ax.set_ylabel('Variance of z', fontweight='bold')
ax.set_title('Variance of Sampled Latent Codes (z) - 30D Model', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.bar(range(len(mean_vars)), mean_vars, color='green', alpha=0.7)
ax.axhline(y=1.0, color='red', linestyle='--', label='Prior variance (1.0)')
ax.set_xlabel('Latent Dimension', fontweight='bold')
ax.set_ylabel('Mean Posterior Variance', fontweight='bold')
ax.set_title('Mean Posterior Variance (σ²) - 30D Model', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
top_dims = np.argsort(mu_vars)[-5:]
bottom_dims = np.argsort(mu_vars)[:5]

positions = []
data_to_plot = []
labels = []
colors = []

for i, dim in enumerate(top_dims):
    positions.append(i)
    data_to_plot.append(mus[:, dim])
    labels.append(f'Dim {dim}\n(active)')
    colors.append('steelblue')

for i, dim in enumerate(bottom_dims):
    positions.append(len(top_dims) + i)
    data_to_plot.append(mus[:, dim])
    labels.append(f'Dim {dim}\n(collapsed)')
    colors.append('red')

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('μ values', fontweight='bold')
ax.set_title('Distribution of μ: Top 5 vs Bottom 5 Dimensions', fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Latent Collapse Analysis: 30D Model', fontweight='bold', fontsize=15)
plt.tight_layout()

save_figure(fig, 'latent_collapse_analysis')
print(f"  Active dimensions: {len(active_dims)}, Collapsed: {len(collapsed_dims)}")

print("\n✓ Figure sets 1-3 complete!")
print("\nContinuing with comprehensive paper figures...")

# Save latent codes for later use
saved_latent_codes = latent_codes
saved_max_vals_array = np.array([test_max_vals[i] for i in range(n_samples)])

print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\nKey Results (30D Model):")
print(f"  • Active latent dimensions: {len(active_dims)}/{MODEL_CONFIG_30D['latent_dim']}")
print(f"  • PCs for 90% variance: {n_90}")
print(f"  • PCs for 95% variance: {n_95}")
print(f"  • PCs for 99% variance: {n_99}")
print("="*80)
