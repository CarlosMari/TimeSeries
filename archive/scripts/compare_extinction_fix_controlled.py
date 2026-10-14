"""
Controlled comparison: Same samples, before and after extinction fix.
This ensures we're measuring the effect of the fix, not random variation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import sys
import warnings
import pickle

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.models.cvae import LSTM_VAE

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'font.family': 'serif',
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# Also load REAL test data for reference
print("Loading real test data...")
with open('data/TEST_FINAL_PROCESSED.pkl', 'rb') as f:
    test_data = pickle.load(f)
test_samples = test_data['data']
test_max_vals = test_data['reconstruction_max_values']

def apply_extinction_threshold(time_series, threshold=0.001, extinction_value=0.0005):
    """Apply extinction threshold."""
    corrected = time_series.copy()
    for sp in range(time_series.shape[1]):
        below_threshold = np.where(time_series[:, sp] < threshold)[0]
        if len(below_threshold) > 0:
            extinction_time = below_threshold[0]
            corrected[extinction_time:, sp] = extinction_value
    return corrected

def test_lv_fit(x_data):
    """Test LV adherence."""
    N = x_data.shape[1]
    r2_scores = []
    for i in range(N):
        x_i_positive = np.maximum(x_data[:, i], 1e-9)
        log_xi = np.log(x_i_positive)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d_log_xi_dt = np.gradient(log_xi)
        model_lr = LinearRegression()
        model_lr.fit(x_data, d_log_xi_dt)
        predicted = model_lr.predict(x_data)
        r2 = r2_score(d_log_xi_dt, predicted)
        r2_scores.append(r2)
    return np.array(r2_scores)

# ============================================================================
# Test REAL data (for reference)
# ============================================================================
print("\nTesting real test data...")
N_REAL = 500
real_r2_scores = []

for i in range(N_REAL):
    if (i + 1) % 100 == 0:
        print(f"  Real data: {i+1}/{N_REAL}...")

    sample_norm = test_samples[i]
    max_vals = test_max_vals[i]
    time_series = sample_norm * max_vals.reshape(-1, 1)
    time_series = time_series.T

    r2_scores = test_lv_fit(time_series)
    real_r2_scores.append(np.mean(r2_scores))

real_r2_scores = np.array(real_r2_scores)

print(f"\nREAL TEST DATA:")
print(f"  Mean R²: {np.mean(real_r2_scores):.4f}")
print(f"  Std R²:  {np.std(real_r2_scores):.4f}")

# ============================================================================
# Generate samples ONCE and test before/after
# ============================================================================
print("\nGenerating samples...")
N_SAMPLES = 500

# Store generated samples
generated_samples = []

torch.manual_seed(42)  # Fixed seed for reproducibility
np.random.seed(42)

with torch.no_grad():
    for i in range(N_SAMPLES):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{N_SAMPLES}...")

        z = torch.randn(1, 30).to(device)
        recon_norm, max_vals_pred = model.decode(z)

        recon_norm_np = recon_norm.cpu().numpy()[0]
        max_vals_np = max_vals_pred.cpu().numpy()[0]

        time_series = recon_norm_np * max_vals_np.reshape(-1, 1)
        time_series = time_series.T

        generated_samples.append(time_series)

print(f"\nStored {len(generated_samples)} generated samples")

# ============================================================================
# Test SAME samples before and after correction
# ============================================================================
print("\nTesting before and after correction on SAME samples...")

r2_before = []
r2_after = []
extinctions_detected = 0

for i, time_series in enumerate(generated_samples):
    if (i + 1) % 100 == 0:
        print(f"  Tested {i+1}/{N_SAMPLES}...")

    # Detect extinction
    has_extinction = False
    for sp in range(7):
        if np.min(time_series[:, sp]) < 0.001:
            has_extinction = True
            extinctions_detected += 1
            break

    # Before correction
    r2_per_species_before = test_lv_fit(time_series)
    r2_before.append(np.mean(r2_per_species_before))

    # After correction
    time_series_corrected = apply_extinction_threshold(time_series, threshold=0.001)
    r2_per_species_after = test_lv_fit(time_series_corrected)
    r2_after.append(np.mean(r2_per_species_after))

r2_before = np.array(r2_before)
r2_after = np.array(r2_after)

print(f"\nExtinction events detected: {extinctions_detected}/{N_SAMPLES} samples ({100*extinctions_detected/N_SAMPLES:.1f}%)")

print("\n" + "="*80)
print("CONTROLLED COMPARISON: Same Samples Before vs After")
print("="*80)

print("\nREAL TEST DATA (Reference):")
print(f"  Mean R²:      {np.mean(real_r2_scores):.4f}")
print(f"  Median R²:    {np.median(real_r2_scores):.4f}")
print(f"  Std R²:       {np.std(real_r2_scores):.4f}")
print(f"  % R² > 0.90:  {100 * np.mean(real_r2_scores > 0.90):.1f}%")

print("\nGENERATED - BEFORE CORRECTION:")
print(f"  Mean R²:      {np.mean(r2_before):.4f}")
print(f"  Median R²:    {np.median(r2_before):.4f}")
print(f"  Std R²:       {np.std(r2_before):.4f}")
print(f"  % R² > 0.90:  {100 * np.mean(r2_before > 0.90):.1f}%")
print(f"  % R² < 0.70:  {100 * np.mean(r2_before < 0.70):.1f}%")

print("\nGENERATED - AFTER CORRECTION:")
print(f"  Mean R²:      {np.mean(r2_after):.4f}")
print(f"  Median R²:    {np.median(r2_after):.4f}")
print(f"  Std R²:       {np.std(r2_after):.4f}")
print(f"  % R² > 0.90:  {100 * np.mean(r2_after > 0.90):.1f}%")
print(f"  % R² < 0.70:  {100 * np.mean(r2_after < 0.70):.1f}%")

print("\nIMPROVEMENT FROM CORRECTION:")
print(f"  Δ Mean R²:        {np.mean(r2_after) - np.mean(r2_before):+.4f}")
print(f"  Δ Median R²:      {np.median(r2_after) - np.median(r2_before):+.4f}")
print(f"  Δ % R² > 0.90:    {100 * (np.mean(r2_after > 0.90) - np.mean(r2_before > 0.90)):+.1f}%")
print(f"  Δ % R² < 0.70:    {100 * (np.mean(r2_after < 0.70) - np.mean(r2_before < 0.70)):+.1f}%")

print("\nGAP TO REAL DATA:")
print(f"  Before: {np.mean(real_r2_scores) - np.mean(r2_before):.4f}")
print(f"  After:  {np.mean(real_r2_scores) - np.mean(r2_after):.4f}")

# ============================================================================
# Create figure
# ============================================================================
print("\nCreating comparison figure...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35,
                       left=0.06, right=0.98, top=0.91, bottom=0.07)

# Panel A: Three-way comparison
ax1 = fig.add_subplot(gs[0, :])
bins = np.linspace(0.5, 1.0, 50)

ax1.hist(real_r2_scores, bins=bins, alpha=0.5, color='green',
         label=f'Real Data (μ={np.mean(real_r2_scores):.3f})',
         edgecolor='black', linewidth=0.5)
ax1.hist(r2_before, bins=bins, alpha=0.5, color='lightcoral',
         label=f'Generated - Before (μ={np.mean(r2_before):.3f})',
         edgecolor='black', linewidth=0.5)
ax1.hist(r2_after, bins=bins, alpha=0.5, color='steelblue',
         label=f'Generated - After (μ={np.mean(r2_after):.3f})',
         edgecolor='black', linewidth=0.5)

ax1.axvline(np.mean(real_r2_scores), color='green', linestyle='--', linewidth=2.5)
ax1.axvline(np.mean(r2_before), color='red', linestyle='--', linewidth=2.5)
ax1.axvline(np.mean(r2_after), color='blue', linestyle='--', linewidth=2.5)

ax1.set_xlabel('Mean R² (LV Adherence)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('A) Three-Way Comparison: Real Data vs Generated (Before vs After Correction)',
              fontweight='bold', loc='left', pad=10)
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, alpha=0.3)

# Panel B: Before vs After scatter
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(r2_before, r2_after, alpha=0.4, s=20, c='steelblue')
ax2.plot([0.5, 1], [0.5, 1], 'k--', linewidth=1.5, alpha=0.5, label='No change')
ax2.set_xlabel('R² Before Correction', fontweight='bold')
ax2.set_ylabel('R² After Correction', fontweight='bold')
ax2.set_title('B) Individual Sample Changes', fontweight='bold', loc='left')
ax2.legend(loc='lower right', frameon=True)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 1.0)
ax2.set_ylim(0.5, 1.0)

# Panel C: Improvement distribution
ax3 = fig.add_subplot(gs[1, 1])
improvement = r2_after - r2_before
ax3.hist(improvement, bins=40, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax3.axvline(np.mean(improvement), color='blue', linestyle='-', linewidth=2.5,
            label=f'Mean = {np.mean(improvement):+.4f}')
ax3.set_xlabel('ΔR² (After - Before)', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('C) Distribution of Improvement', fontweight='bold', loc='left')
ax3.legend(loc='upper right', frameon=True)
ax3.grid(True, alpha=0.3)

# Panel D: Box plot comparison
ax4 = fig.add_subplot(gs[1, 2])
box_data = [real_r2_scores, r2_before, r2_after]
bp = ax4.boxplot(box_data, labels=['Real', 'Gen\nBefore', 'Gen\nAfter'],
                  patch_artist=True, widths=0.6)
colors_box = ['lightgreen', 'lightcoral', 'lightblue']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax4.set_ylabel('R²', fontweight='bold')
ax4.set_title('D) Distribution Comparison', fontweight='bold', loc='left')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(0.9, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)

# Panels E-G: Example trajectories
colors_sp = plt.cm.tab10(np.linspace(0, 0.7, 7))
time = np.arange(65)

# Find a sample with good improvement
improvements_individual = r2_after - r2_before
best_improvement_idx = np.argmax(improvements_individual)
example_sample = generated_samples[best_improvement_idx]
example_corrected = apply_extinction_threshold(example_sample, threshold=0.001)

ax5 = fig.add_subplot(gs[2, 0])
for i in range(7):
    ax5.plot(time, example_sample[:, i], color=colors_sp[i], linewidth=1.5, alpha=0.8)
ax5.axhline(0.001, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax5.set_xlabel('Time', fontweight='bold')
ax5.set_ylabel('Abundance', fontweight='bold')
ax5.set_title(f'E) Before (R² = {r2_before[best_improvement_idx]:.3f})',
              fontweight='bold', loc='left')
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
for i in range(7):
    ax6.plot(time, example_corrected[:, i], color=colors_sp[i], linewidth=1.5,
             alpha=0.8, label=f'Sp{i+1}')
ax6.axhline(0.001, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax6.set_xlabel('Time', fontweight='bold')
ax6.set_ylabel('Abundance', fontweight='bold')
ax6.set_title(f'F) After (R² = {r2_after[best_improvement_idx]:.3f})',
              fontweight='bold', loc='left')
ax6.legend(loc='upper right', frameon=True, fontsize=6, ncol=2)
ax6.grid(True, alpha=0.3)

# Panel G: Per-species R² comparison
ax7 = fig.add_subplot(gs[2, 2])
r2_before_example = test_lv_fit(example_sample)
r2_after_example = test_lv_fit(example_corrected)
x_pos = np.arange(7)
width = 0.35
ax7.bar(x_pos - width/2, r2_before_example, width, label='Before',
        color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
ax7.bar(x_pos + width/2, r2_after_example, width, label='After',
        color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
ax7.axhline(0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax7.set_xlabel('Species', fontweight='bold')
ax7.set_ylabel('R²', fontweight='bold')
ax7.set_title('G) Per-Species Improvement', fontweight='bold', loc='left')
ax7.set_xticks(x_pos)
ax7.set_xticklabels([f'{i+1}' for i in range(7)])
ax7.legend(loc='lower right', frameon=True)
ax7.grid(True, alpha=0.3, axis='y')

fig.suptitle('Extinction Threshold Fix: Controlled Comparison on Same Samples',
             fontsize=12, fontweight='bold')

output_png = 'final figures/fig_extinction_fix_controlled.png'
output_pdf = 'final figures/fig_extinction_fix_controlled.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\nFigure saved:")
print(f"  {output_png}")
print(f"  {output_pdf}")

plt.close()

print("\n" + "="*80)
print("✓ Controlled comparison complete!")
print("="*80)
