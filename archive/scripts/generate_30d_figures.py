"""
Comprehensive figure generation for 30D latent model.
All figures saved with "_30" suffix to avoid overwriting 50D results.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fft, fftfreq
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE

# ====================================================================================
# CONFIGURATION FOR 30D MODEL
# ====================================================================================
model_config = {
    "in_channels": 7,
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,  # ← CHANGED FROM 50
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'
TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
OUTPUT_DIR = 'final figures/'
OUTPUT_SUFFIX = '_30'

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

print("="*80)
print("FIGURE GENERATION: 30D LATENT MODEL")
print("="*80)
print(f"Model: {MODEL_PATH}")
print(f"Latent dim: {model_config['latent_dim']}")
print(f"Output suffix: {OUTPUT_SUFFIX}")
print("="*80)

# ====================================================================================
# LOAD MODEL AND DATA
# ====================================================================================
def load_model_and_data():
    """Load 30D model and test data."""
    print("\nLoading model and data...")

    model = LSTM_VAE(config=model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)

    print(f"✓ Model loaded (latent_dim={model_config['latent_dim']})")
    print(f"✓ Test samples: {len(data_package['data'])}")

    return model, data_package['data'], data_package['reconstruction_max_values']

model, test_data, test_max_vals = load_model_and_data()

# ====================================================================================
# FIGURE 1: RECONSTRUCTION QUALITY
# ====================================================================================
print("\n[1/7] Reconstruction Quality Analysis...")

n_eval = min(1000, len(test_data))
recon_r2_list = []
max_val_r2_list = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(n_eval), desc="  Evaluating"):
        x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        max_vals_true = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        recon_norm, _, _, max_vals_pred, _ = model(x_norm, max_vals_true, teacher_forcing_ratio=0.0)

        # R² for reconstruction
        recon_true = x_norm.cpu().numpy().flatten()
        recon_pred = recon_norm.cpu().numpy().flatten()
        ss_res = np.sum((recon_true - recon_pred)**2)
        ss_tot = np.sum((recon_true - np.mean(recon_true))**2)
        r2_recon = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        recon_r2_list.append(r2_recon)

        # R² for max values
        max_true = max_vals_true.cpu().numpy().flatten()
        max_pred = max_vals_pred.cpu().numpy().flatten()
        ss_res = np.sum((max_true - max_pred)**2)
        ss_tot = np.sum((max_true - np.mean(max_true))**2)
        r2_max = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        max_val_r2_list.append(r2_max)

avg_recon_r2 = np.mean(recon_r2_list)
avg_max_r2 = np.mean(max_val_r2_list)

print(f"  ✓ Reconstruction R²: {avg_recon_r2:.4f}")
print(f"  ✓ Max Value R²: {avg_max_r2:.4f}")

# Save metrics
with open(f'{OUTPUT_DIR}metrics{OUTPUT_SUFFIX}.txt', 'w') as f:
    f.write(f"30D Latent Model Performance\n")
    f.write(f"="*50 + "\n")
    f.write(f"Reconstruction R²: {avg_recon_r2:.4f}\n")
    f.write(f"Max Value R²: {avg_max_r2:.4f}\n")
    f.write(f"Evaluated on {n_eval} samples\n")

# ====================================================================================
# FIGURE 2: VARIANCE EXPLAINED (PCA)
# ====================================================================================
print("\n[2/7] Variance Explained (PCA)...")

# Encode samples
n_encode = min(5000, len(test_data))
latent_codes = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(n_encode), desc="  Encoding"):
        x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        x_rnn = x_norm.permute(0, 2, 1)
        _, (h_n, _) = model.encoder_rnn(x_rnn)
        encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

        z_shape = model.shape_projector(encoded_summary)
        z_scale = model.scale_encoder(max_vals)
        bottleneck_input = torch.cat([z_shape, z_scale], dim=1)

        mu = model.fc_mu(bottleneck_input)
        latent_codes.append(mu.cpu().numpy()[0])

latent_codes = np.array(latent_codes)

# PCA
pca = PCA()
pca.fit(latent_codes)
var_ratio = pca.explained_variance_ratio_
cumsum = np.cumsum(var_ratio)

n_90 = np.argmax(cumsum >= 0.90) + 1
n_95 = np.argmax(cumsum >= 0.95) + 1
n_99 = np.argmax(cumsum >= 0.99) + 1

print(f"  ✓ PCs for 90%: {n_90}/{len(var_ratio)}")
print(f"  ✓ PCs for 95%: {n_95}/{len(var_ratio)}")
print(f"  ✓ PCs for 99%: {n_99}/{len(var_ratio)}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Full spectrum
x_pos = np.arange(len(var_ratio))
ax1.bar(x_pos, var_ratio * 100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)

ax1_twin = ax1.twinx()
ax1_twin.plot(x_pos, cumsum * 100, 'ro-', linewidth=2.5, markersize=4, alpha=0.8)
ax1_twin.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='90%')
ax1_twin.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='95%')
ax1_twin.axhline(y=99, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='99%')

ax1.set_xlabel('Principal Component', fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
ax1_twin.set_ylabel('Cumulative (%)', fontweight='bold', color='red')
ax1.set_title(f'PCA Variance Explained (30D Latent Space)', fontweight='bold', fontsize=13)
ax1.grid(alpha=0.3, axis='y')
ax1_twin.set_ylim([0, 105])
ax1_twin.legend(loc='right')

# Detail view
n_show = min(20, len(var_ratio))
x_pos_zoom = np.arange(n_show)
bars2 = ax2.bar(x_pos_zoom, var_ratio[:n_show] * 100, alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.8)

ax2_twin = ax2.twinx()
ax2_twin.plot(x_pos_zoom, cumsum[:n_show] * 100, 'ro-', linewidth=2.5, markersize=6, alpha=0.8)

for i in range(min(5, n_show)):
    bars2[i].set_color('darkblue')
    bars2[i].set_alpha(0.9)

for i, (bar, val) in enumerate(zip(bars2, var_ratio[:n_show] * 100)):
    if i < 10:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Principal Component', fontweight='bold')
ax2.set_ylabel('Variance Explained (%)', fontweight='bold')
ax2_twin.set_ylabel('Cumulative (%)', fontweight='bold', color='red')
ax2.set_title(f'First {n_show} Components (Detail)', fontweight='bold', fontsize=13)
ax2.grid(alpha=0.3, axis='y')
ax2_twin.set_ylim([0, 105])

plt.suptitle('Variance Explained: 30D Model', fontweight='bold', fontsize=15)
plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}fig_variance_explained{OUTPUT_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}fig_variance_explained{OUTPUT_SUFFIX}.pdf', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: fig_variance_explained{OUTPUT_SUFFIX}.png/pdf")
plt.close()

# ====================================================================================
# FIGURE 3: LATENT COLLAPSE ANALYSIS
# ====================================================================================
print("\n[3/7] Latent Collapse Analysis...")

mus = []
log_vars = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(min(5000, len(test_data))), desc="  Analyzing collapse"):
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

        mus.append(mu.cpu().numpy()[0])
        log_vars.append(log_var.cpu().numpy()[0])

mus = np.array(mus)
mu_vars = np.var(mus, axis=0)

threshold = 0.01
collapsed = np.sum(mu_vars < threshold)
active = np.sum(mu_vars >= threshold)

print(f"  ✓ Active dimensions: {active}/{len(mu_vars)}")
print(f"  ✓ Collapsed dimensions: {collapsed}/{len(mu_vars)}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(len(mu_vars)), mu_vars, color='steelblue', alpha=0.7)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
          label=f'Collapse threshold ({threshold})')
ax.set_xlabel('Latent Dimension', fontweight='bold')
ax.set_ylabel('Variance of μ', fontweight='bold')
ax.set_title(f'Latent Dimension Utilization (30D Model): {active} Active, {collapsed} Collapsed',
            fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}latent_collapse_analysis{OUTPUT_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}latent_collapse_analysis{OUTPUT_SUFFIX}.pdf', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: latent_collapse_analysis{OUTPUT_SUFFIX}.png/pdf")
plt.close()

# ====================================================================================
# FIGURE 4: CONTROLLED GENERATION EXAMPLES
# ====================================================================================
print("\n[4/7] Controlled Generation Examples...")

n_gen = 100
gen_data = []
gen_max_vals = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(n_gen), desc="  Generating"):
        z = torch.randn(1, model_config['latent_dim']).to(DEVICE)
        recon_norm, max_vals_pred = model.decode(z)

        gen_data.append(recon_norm.cpu().numpy()[0])
        gen_max_vals.append(max_vals_pred.cpu().numpy()[0])

# Select diverse samples
max_val_sums = [np.sum(m) for m in gen_max_vals]
sorted_indices = np.argsort(max_val_sums)
selected = [sorted_indices[int(i)] for i in np.linspace(0, len(sorted_indices)-1, 6)]

# Plot
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, sample_idx in enumerate(selected):
    ax = axes[idx]
    data_norm = gen_data[sample_idx]
    max_vals = gen_max_vals[sample_idx]
    data_orig = data_norm * max_vals.reshape(1, -1, 1)

    time = np.arange(65)
    for sp in range(7):
        ax.plot(time, data_orig[0, sp, :], color=COLORS[sp], linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Population', fontweight='bold')
    ax.set_title(f'Sample {idx+1} (Scale: {np.mean(max_vals):.1f})', fontweight='bold')
    ax.grid(alpha=0.3)

plt.suptitle('Generated Dynamics (30D Model)', fontweight='bold', fontsize=15)
plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}fig_controlled_generation{OUTPUT_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}fig_controlled_generation{OUTPUT_SUFFIX}.pdf', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: fig_controlled_generation{OUTPUT_SUFFIX}.png/pdf")
plt.close()

# ====================================================================================
# FIGURE 5: LATENT SPACE STRUCTURE (t-SNE & UMAP)
# ====================================================================================
print("\n[5/7] Latent Space Structure (t-SNE & UMAP)...")

# Use already encoded latent codes
total_scales = np.array([np.sum(test_max_vals[i]) for i in range(n_encode)])

print("  Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_tsne = tsne.fit_transform(latent_codes)

print("  Computing UMAP...")
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
latent_umap = umap_reducer.fit_transform(latent_codes)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = ax1.scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                      c=total_scales, cmap='viridis', s=10, alpha=0.6)
ax1.set_xlabel('t-SNE Dim 1', fontweight='bold')
ax1.set_ylabel('t-SNE Dim 2', fontweight='bold')
ax1.set_title('t-SNE (30D Latent)', fontweight='bold', fontsize=13)
plt.colorbar(scatter1, ax=ax1, label='Total Scale')
ax1.grid(alpha=0.3)

scatter2 = ax2.scatter(latent_umap[:, 0], latent_umap[:, 1],
                      c=total_scales, cmap='viridis', s=10, alpha=0.6)
ax2.set_xlabel('UMAP Dim 1', fontweight='bold')
ax2.set_ylabel('UMAP Dim 2', fontweight='bold')
ax2.set_title('UMAP (30D Latent)', fontweight='bold', fontsize=13)
plt.colorbar(scatter2, ax=ax2, label='Total Scale')
ax2.grid(alpha=0.3)

plt.suptitle('Latent Space Structure: 30D Model', fontweight='bold', fontsize=15)
plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}fig_latent_space_structure{OUTPUT_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}fig_latent_space_structure{OUTPUT_SUFFIX}.pdf', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: fig_latent_space_structure{OUTPUT_SUFFIX}.png/pdf")
plt.close()

# ====================================================================================
# FIGURE 6: SCALE CONTROL DEMONSTRATION
# ====================================================================================
print("\n[6/7] Scale Control Demonstration...")

model.eval()
with torch.no_grad():
    z = torch.randn(1, model_config['latent_dim']).to(DEVICE)
    recon_norm, max_vals_pred = model.decode(z)

    recon_norm_np = recon_norm.cpu().numpy()[0]
    max_vals_np = max_vals_pred.cpu().numpy()[0]

scale_factors = [0.5, 1.0, 2.0, 3.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

time = np.arange(65)

for idx, scale in enumerate(scale_factors):
    ax = axes[idx]
    scaled_max = max_vals_np * scale
    data_scaled = recon_norm_np * scaled_max.reshape(1, -1, 1)

    for sp in range(7):
        ax.plot(time, data_scaled[0, sp, :], color=COLORS[sp], linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Population', fontweight='bold')
    ax.set_title(f'Scale: {scale}x (Mean: {np.mean(scaled_max):.1f})', fontweight='bold')
    ax.grid(alpha=0.3)

plt.suptitle('Scale Control: Same Shape, Different Scales (30D Model)',
            fontweight='bold', fontsize=15)
plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}fig_scale_control{OUTPUT_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}fig_scale_control{OUTPUT_SUFFIX}.pdf', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: fig_scale_control{OUTPUT_SUFFIX}.png/pdf")
plt.close()

# ====================================================================================
# FIGURE 7: LATENT INTERPOLATION
# ====================================================================================
print("\n[7/7] Latent Interpolation...")

# Get two diverse latent codes
z1_idx = np.argmin([np.sum(test_max_vals[i]) for i in range(min(1000, len(test_data)))])
z2_idx = np.argmax([np.sum(test_max_vals[i]) for i in range(min(1000, len(test_data)))])

z1 = latent_codes[z1_idx]
z2 = latent_codes[z2_idx]

alphas = np.linspace(0, 1, 6)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

time = np.arange(65)

model.eval()
with torch.no_grad():
    for idx, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z1 + alpha * z2
        z_torch = torch.tensor(z_interp, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        recon_norm, max_vals_pred = model.decode(z_torch)

        recon_np = recon_norm.cpu().numpy()[0]
        max_np = max_vals_pred.cpu().numpy()[0]
        data_orig = recon_np * max_np.reshape(1, -1, 1)

        ax = axes[idx]
        for sp in range(7):
            ax.plot(time, data_orig[0, sp, :], color=COLORS[sp], linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Population', fontweight='bold')
        ax.set_title(f'α = {alpha:.2f}', fontweight='bold')
        ax.grid(alpha=0.3)

plt.suptitle('Latent Interpolation (30D Model)', fontweight='bold', fontsize=15)
plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}fig_latent_interpolation{OUTPUT_SUFFIX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}fig_latent_interpolation{OUTPUT_SUFFIX}.pdf', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: fig_latent_interpolation{OUTPUT_SUFFIX}.png/pdf")
plt.close()

# ====================================================================================
# SUMMARY
# ====================================================================================
print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\n30D Model Performance:")
print(f"  • Reconstruction R²: {avg_recon_r2:.4f}")
print(f"  • Max Value R²: {avg_max_r2:.4f}")
print(f"  • Active dimensions: {active}/{model_config['latent_dim']}")
print(f"  • PCs for 90%: {n_90}, 95%: {n_95}, 99%: {n_99}")
print(f"\nFigures saved with '{OUTPUT_SUFFIX}' suffix in: {OUTPUT_DIR}")
print("="*80)
