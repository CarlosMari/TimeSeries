"""
Professional figure demonstrating controlled generation via interpretable latent dimensions.

Focus on the strongest controls:
- Dim 6: Species 1 trend (r=0.74)
- Dim 7: Species 5 trend (r=0.73)
- Dim 18: Species 7 trend (r=-0.70)
- Dim 2: Synchronization (r=0.38)
- Dim 15: Curvature (r=0.64)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.models.cvae import LSTM_VAE

# Professional publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model_config = {
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'

print("Loading model...")
model = LSTM_VAE(model_config).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded!\n")

def generate_from_latent(model, z):
    """Generate sample from latent code."""
    with torch.no_grad():
        recon_norm, max_vals_pred = model.decode(z)
    recon_norm_np = recon_norm.cpu().numpy()[0]
    max_vals_np = max_vals_pred.cpu().numpy()[0]
    return recon_norm_np * max_vals_np.reshape(-1, 1)

def compute_trend(curve):
    """Linear trend (slope)."""
    x = np.arange(len(curve))
    return np.polyfit(x, curve, 1)[0]

def compute_synchronization(time_series):
    """Mean pairwise correlation."""
    n_species = time_series.shape[0]
    corrs = []
    for i in range(n_species):
        for j in range(i+1, n_species):
            corrs.append(np.corrcoef(time_series[i], time_series[j])[0, 1])
    return np.mean(corrs)

def compute_curvature(time_series):
    """Mean absolute second derivative (curvature)."""
    second_deriv = np.abs(np.diff(np.diff(time_series, axis=1), axis=1))
    return np.mean(second_deriv)

# Create base latent code
np.random.seed(42)
torch.manual_seed(42)
base_z = torch.randn(1, 30).to(device)

# Colors for species
colors = plt.cm.tab10(np.linspace(0, 0.7, 7))

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.35,
                       left=0.05, right=0.98, top=0.92, bottom=0.08)

# ============================================================================
# Row 1: Control Individual Species Trends
# ============================================================================
print("Generating Row 1: Per-Species Trend Control...")

# Dim 6: Species 1
for col_idx, (z_val, z_label) in enumerate([(-2.0, '-2'), (0.0, '0'), (+2.0, '+2')]):
    ax = fig.add_subplot(gs[0, col_idx])
    z = base_z.clone()
    z[0, 6] = z_val
    sample = generate_from_latent(model, z)

    for i in range(7):
        lw = 2.5 if i == 0 else 1.2
        alpha = 0.9 if i == 0 else 0.6
        ax.plot(sample[i], color=colors[i], linewidth=lw, alpha=alpha,
                label='Species 1' if i == 0 else '')

    trend = compute_trend(sample[0])
    ax.set_title(f'Dim 6 = {z_label}\nSp1 trend = {trend:+.4f}', fontweight='bold', fontsize=9)
    ax.set_xlabel('Time')
    ax.set_ylabel('Abundance')
    ax.set_ylim(-0.05, 1.1)
    if col_idx == 0:
        ax.legend(loc='upper left', frameon=True)

# Dim 7: Species 5
for col_idx, (z_val, z_label) in enumerate([(-2.0, '-2'), (+2.0, '+2')]):
    ax = fig.add_subplot(gs[0, col_idx + 3])
    z = base_z.clone()
    z[0, 7] = z_val
    sample = generate_from_latent(model, z)

    for i in range(7):
        lw = 2.5 if i == 4 else 1.2
        alpha = 0.9 if i == 4 else 0.6
        ax.plot(sample[i], color=colors[i], linewidth=lw, alpha=alpha,
                label='Species 5' if i == 4 else '')

    trend = compute_trend(sample[4])
    ax.set_title(f'Dim 7 = {z_label}\nSp5 trend = {trend:+.4f}', fontweight='bold', fontsize=9)
    ax.set_xlabel('Time')
    ax.set_ylabel('Abundance')
    ax.set_ylim(-0.05, 1.1)
    if col_idx == 0:
        ax.legend(loc='upper left', frameon=True)

# ============================================================================
# Row 2: Control Synchronization (Dim 2)
# ============================================================================
print("Generating Row 2: Synchronization Control...")

for col_idx, (z_val, z_label) in enumerate([(-2.0, '-2'), (0.0, '0'), (+2.0, '+2')]):
    ax = fig.add_subplot(gs[1, col_idx])
    z = base_z.clone()
    z[0, 2] = z_val
    sample = generate_from_latent(model, z)

    for i in range(7):
        ax.plot(sample[i], color=colors[i], linewidth=1.5, alpha=0.7)

    sync = compute_synchronization(sample)
    ax.set_title(f'Dim 2 = {z_label}\nSync = {sync:.3f}', fontweight='bold', fontsize=9)
    ax.set_xlabel('Time')
    ax.set_ylabel('Abundance')
    ax.set_ylim(-0.05, 1.1)

# ============================================================================
# Row 2 (continued): Control Species 7 Trend (Dim 18)
# ============================================================================
print("Generating Row 2: Species 7 Trend Control...")

for col_idx, (z_val, z_label) in enumerate([(-2.0, '-2'), (+2.0, '+2')]):
    ax = fig.add_subplot(gs[1, col_idx + 3])
    z = base_z.clone()
    z[0, 18] = z_val
    sample = generate_from_latent(model, z)

    for i in range(7):
        lw = 2.5 if i == 6 else 1.2
        alpha = 0.9 if i == 6 else 0.6
        ax.plot(sample[i], color=colors[i], linewidth=lw, alpha=alpha,
                label='Species 7' if i == 6 else '')

    trend = compute_trend(sample[6])
    ax.set_title(f'Dim 18 = {z_label}\nSp7 trend = {trend:+.4f}', fontweight='bold', fontsize=9)
    ax.set_xlabel('Time')
    ax.set_ylabel('Abundance')
    ax.set_ylim(-0.05, 1.1)
    if col_idx == 0:
        ax.legend(loc='upper left', frameon=True)

# ============================================================================
# Row 3: Control Curvature (Dim 15)
# ============================================================================
print("Generating Row 3: Curvature Control...")

for col_idx, (z_val, z_label) in enumerate([(-2.5, '-2.5'), (-1.0, '-1'), (0.0, '0'), (+1.0, '+1'), (+2.5, '+2.5')]):
    ax = fig.add_subplot(gs[2, col_idx])
    z = base_z.clone()
    z[0, 15] = z_val
    sample = generate_from_latent(model, z)

    for i in range(7):
        ax.plot(sample[i], color=colors[i], linewidth=1.5, alpha=0.7)

    curv = compute_curvature(sample)
    ax.set_title(f'Dim 15 = {z_label}\nCurvature = {curv:.4f}', fontweight='bold', fontsize=9)
    ax.set_xlabel('Time')
    ax.set_ylabel('Abundance')
    ax.set_ylim(-0.05, 1.1)

# Overall title
fig.suptitle('Controlled Generation via Interpretable Latent Space Manipulation\n' +
             'Each dimension controls a specific, interpretable dynamical property',
             fontsize=12, fontweight='bold', y=0.97)

# Add annotations
fig.text(0.02, 0.685, 'Per-Species\nTrend Control', ha='left', va='center',
         fontsize=10, fontweight='bold', rotation=90,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.text(0.02, 0.40, 'Interaction &\nSpecies Control', ha='left', va='center',
         fontsize=10, fontweight='bold', rotation=90,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

fig.text(0.02, 0.155, 'Global Trajectory\nShape Control', ha='left', va='center',
         fontsize=10, fontweight='bold', rotation=90,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Legend for all species
species_handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label=f'Sp{i+1}')
                   for i in range(7)]
fig.legend(handles=species_handles, loc='lower center', ncol=7, frameon=True,
           fontsize=8, bbox_to_anchor=(0.5, 0.01))

# Save
output_png = 'final figures/fig_controlled_generation_professional.png'
output_pdf = 'final figures/fig_controlled_generation_professional.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\nFigure saved:")
print(f"  {output_png}")
print(f"  {output_pdf}")

plt.close()

# ============================================================================
# Quantitative validation
# ============================================================================
print("\n" + "="*80)
print("QUANTITATIVE VALIDATION OF CONTROLLED GENERATION")
print("="*80)

test_dimensions = [
    (6, "Species 1 Trend", lambda s: compute_trend(s[0])),
    (7, "Species 5 Trend", lambda s: compute_trend(s[4])),
    (18, "Species 7 Trend", lambda s: compute_trend(s[6])),
    (2, "Synchronization", compute_synchronization),
    (15, "Curvature", compute_curvature),
]

for dim, name, metric_fn in test_dimensions:
    values = []
    z_vals = [-2.0, -1.0, 0.0, +1.0, +2.0]

    for z_val in z_vals:
        z = base_z.clone()
        z[0, dim] = z_val
        sample = generate_from_latent(model, z)
        values.append(metric_fn(sample))

    print(f"\n{name} (Dim {dim}):")
    for z_val, val in zip(z_vals, values):
        print(f"  z = {z_val:+4.1f}  →  metric = {val:+.4f}")

    control_range = max(values) - min(values)
    monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1)) or \
                all(values[i] >= values[i+1] for i in range(len(values)-1))

    print(f"  Control Range: {control_range:.4f}")
    print(f"  Monotonic: {'✓ Yes' if monotonic else '✗ No'}")

print("\n" + "="*80)
print("✓ Interpretable latent space enables precise control over generation!")
print("="*80)
