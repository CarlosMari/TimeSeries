"""
Parameter sweep: Find optimal extinction threshold value.

Tests multiple threshold values to determine which gives best LV adherence.
Uses large sample size for robust statistics.
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.models.cvae import LSTM_VAE

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'font.family': 'serif',
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
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

def apply_extinction_threshold(time_series, threshold, extinction_value=None):
    """Apply extinction threshold with specified value."""
    if extinction_value is None:
        extinction_value = threshold / 2  # Default: half of threshold

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
# Generate samples once (use same samples for all thresholds)
# ============================================================================
N_SAMPLES = 2000  # Larger sample size for robust statistics
print(f"Generating {N_SAMPLES} samples...")

generated_samples = []

torch.manual_seed(42)
np.random.seed(42)

with torch.no_grad():
    for i in tqdm(range(N_SAMPLES), desc="Generating"):
        z = torch.randn(1, 30).to(device)
        recon_norm, max_vals_pred = model.decode(z)

        recon_norm_np = recon_norm.cpu().numpy()[0]
        max_vals_np = max_vals_pred.cpu().numpy()[0]

        time_series = recon_norm_np * max_vals_np.reshape(-1, 1)
        time_series = time_series.T

        generated_samples.append(time_series)

print(f"Generated {len(generated_samples)} samples\n")

# ============================================================================
# Sweep over threshold values
# ============================================================================
# Test thresholds from 0 (no correction) to 0.1
thresholds_to_test = [
    0.0,      # No correction (baseline)
    0.0001,   # Very strict
    0.0005,
    0.001,    # Current default
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.1,      # Very aggressive
]

print("="*80)
print("THRESHOLD SWEEP ANALYSIS")
print("="*80)
print(f"\nTesting {len(thresholds_to_test)} threshold values on {N_SAMPLES} samples...")
print()

results = {}

for threshold in thresholds_to_test:
    print(f"Testing threshold = {threshold:.4f}...")

    r2_scores = []
    extinctions_applied = 0

    for i, time_series in enumerate(tqdm(generated_samples, desc=f"  θ={threshold:.4f}", leave=False)):
        # Apply correction
        if threshold == 0.0:
            time_series_corrected = time_series  # No correction
        else:
            time_series_corrected = apply_extinction_threshold(time_series, threshold)
            if not np.allclose(time_series, time_series_corrected):
                extinctions_applied += 1

        # Test LV fit
        r2_per_species = test_lv_fit(time_series_corrected)
        r2_scores.append(np.mean(r2_per_species))

    r2_scores = np.array(r2_scores)

    results[threshold] = {
        'r2_mean': np.mean(r2_scores),
        'r2_median': np.median(r2_scores),
        'r2_std': np.std(r2_scores),
        'r2_min': np.min(r2_scores),
        'r2_max': np.max(r2_scores),
        'pct_above_90': 100 * np.mean(r2_scores > 0.90),
        'pct_above_95': 100 * np.mean(r2_scores > 0.95),
        'pct_below_70': 100 * np.mean(r2_scores < 0.70),
        'extinctions_applied': extinctions_applied,
        'r2_scores': r2_scores,
    }

    print(f"  Mean R²: {results[threshold]['r2_mean']:.4f}, "
          f"% > 0.90: {results[threshold]['pct_above_90']:.1f}%, "
          f"Extinctions: {extinctions_applied}/{N_SAMPLES}\n")

# ============================================================================
# Find optimal threshold
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Baseline (no correction)
baseline_r2 = results[0.0]['r2_mean']
print(f"\nBaseline (no correction): Mean R² = {baseline_r2:.4f}")

print("\nAll thresholds:")
print(f"{'Threshold':<12} {'Mean R²':<10} {'Δ vs Base':<12} {'% > 0.90':<10} {'% < 0.70':<10} {'Extinctions':<12}")
print("-" * 80)

for threshold in thresholds_to_test:
    r = results[threshold]
    delta = r['r2_mean'] - baseline_r2
    print(f"{threshold:<12.4f} {r['r2_mean']:<10.4f} {delta:+<12.4f} "
          f"{r['pct_above_90']:<10.1f} {r['pct_below_70']:<10.1f} "
          f"{r['extinctions_applied']:<12d}")

# Find best threshold (maximize mean R²)
best_threshold_mean = max(results.keys(), key=lambda t: results[t]['r2_mean'])
best_r2_mean = results[best_threshold_mean]['r2_mean']

# Find threshold that maximizes % > 0.90
best_threshold_quality = max(results.keys(), key=lambda t: results[t]['pct_above_90'])
best_pct_90 = results[best_threshold_quality]['pct_above_90']

print("\n" + "="*80)
print("OPTIMAL THRESHOLDS")
print("="*80)
print(f"\nBest for Mean R²:         θ = {best_threshold_mean:.4f} (R² = {best_r2_mean:.4f})")
print(f"Best for % High Quality:  θ = {best_threshold_quality:.4f} (% > 0.90 = {best_pct_90:.1f}%)")

# Recommend based on multiple criteria
# Good threshold should: maximize R², maximize % > 0.90, minimize % < 0.70
scores = {}
for threshold in thresholds_to_test:
    if threshold == 0.0:
        continue  # Skip baseline
    r = results[threshold]
    # Composite score: weighted combination
    score = (r['r2_mean'] - baseline_r2) * 100  # Improvement in R²
    score += r['pct_above_90'] / 100  # % high quality
    score -= r['pct_below_70'] * 10  # Penalty for failures
    scores[threshold] = score

best_overall = max(scores.keys(), key=lambda t: scores[t])
print(f"\nRecommended (composite):  θ = {best_overall:.4f} (score = {scores[best_overall]:.2f})")

# ============================================================================
# Create comprehensive visualization
# ============================================================================
print("\nCreating visualization...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                       left=0.06, right=0.98, top=0.93, bottom=0.07)

# Panel A: Mean R² vs Threshold
ax1 = fig.add_subplot(gs[0, 0])
thresholds_plot = list(results.keys())
mean_r2_plot = [results[t]['r2_mean'] for t in thresholds_plot]
ax1.plot(thresholds_plot, mean_r2_plot, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(baseline_r2, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline')
ax1.axvline(best_threshold_mean, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'Best (θ={best_threshold_mean:.4f})')
ax1.set_xlabel('Extinction Threshold', fontweight='bold')
ax1.set_ylabel('Mean R²', fontweight='bold')
ax1.set_title('A) Mean R² vs Threshold', fontweight='bold', loc='left')
ax1.legend(loc='best', frameon=True)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Panel B: % High Quality vs Threshold
ax2 = fig.add_subplot(gs[0, 1])
pct_90_plot = [results[t]['pct_above_90'] for t in thresholds_plot]
ax2.plot(thresholds_plot, pct_90_plot, 'o-', linewidth=2, markersize=8, color='green')
ax2.axhline(results[0.0]['pct_above_90'], color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.axvline(best_threshold_quality, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Extinction Threshold', fontweight='bold')
ax2.set_ylabel('% Samples with R² > 0.90', fontweight='bold')
ax2.set_title('B) High-Quality Sample Rate vs Threshold', fontweight='bold', loc='left')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# Panel C: Std R² vs Threshold
ax3 = fig.add_subplot(gs[0, 2])
std_r2_plot = [results[t]['r2_std'] for t in thresholds_plot]
ax3.plot(thresholds_plot, std_r2_plot, 'o-', linewidth=2, markersize=8, color='purple')
ax3.axhline(results[0.0]['r2_std'], color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax3.set_xlabel('Extinction Threshold', fontweight='bold')
ax3.set_ylabel('Std(R²)', fontweight='bold')
ax3.set_title('C) R² Variability vs Threshold', fontweight='bold', loc='left')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')

# Panel D: Distribution comparison (key thresholds)
ax4 = fig.add_subplot(gs[1, :])
key_thresholds = [0.0, 0.001, 0.01, 0.05]
colors_dist = ['red', 'green', 'blue', 'orange']
bins = np.linspace(0.5, 1.0, 50)

for threshold, color in zip(key_thresholds, colors_dist):
    r2_data = results[threshold]['r2_scores']
    label = f'θ={threshold:.4f} (μ={results[threshold]["r2_mean"]:.3f})'
    ax4.hist(r2_data, bins=bins, alpha=0.4, color=color, label=label, edgecolor='black', linewidth=0.5)

ax4.set_xlabel('R²', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.set_title('D) R² Distributions for Key Thresholds', fontweight='bold', loc='left')
ax4.legend(loc='upper left', frameon=True)
ax4.grid(True, alpha=0.3)

# Panel E: Box plot comparison
ax5 = fig.add_subplot(gs[2, 0])
box_data = [results[t]['r2_scores'] for t in key_thresholds]
bp = ax5.boxplot(box_data, tick_labels=[f'{t:.4f}' for t in key_thresholds],
                  patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors_dist):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax5.set_xlabel('Threshold', fontweight='bold')
ax5.set_ylabel('R²', fontweight='bold')
ax5.set_title('E) Distribution Comparison', fontweight='bold', loc='left')
ax5.axhline(0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax5.grid(True, alpha=0.3, axis='y')

# Panel F: Failure rate vs Threshold
ax6 = fig.add_subplot(gs[2, 1])
pct_below_70 = [results[t]['pct_below_70'] for t in thresholds_plot]
ax6.plot(thresholds_plot, pct_below_70, 'o-', linewidth=2, markersize=8, color='red')
ax6.set_xlabel('Extinction Threshold', fontweight='bold')
ax6.set_ylabel('% Samples with R² < 0.70', fontweight='bold')
ax6.set_title('F) Failure Rate vs Threshold', fontweight='bold', loc='left')
ax6.grid(True, alpha=0.3)
ax6.set_xscale('log')

# Panel G: Improvement over baseline
ax7 = fig.add_subplot(gs[2, 2])
improvements = [(results[t]['r2_mean'] - baseline_r2) * 100 for t in thresholds_plot]
colors_bar = ['red' if imp < 0 else 'green' for imp in improvements]
bars = ax7.bar(range(len(thresholds_plot)), improvements, color=colors_bar, alpha=0.7, edgecolor='black')
ax7.axhline(0, color='black', linestyle='-', linewidth=1.5)
ax7.set_xlabel('Threshold Index', fontweight='bold')
ax7.set_ylabel('ΔR² vs Baseline (%)', fontweight='bold')
ax7.set_title('G) Improvement over Baseline', fontweight='bold', loc='left')
ax7.set_xticks(range(len(thresholds_plot)))
ax7.set_xticklabels([f'{t:.4f}' for t in thresholds_plot], rotation=45, ha='right', fontsize=7)
ax7.grid(True, alpha=0.3, axis='y')

# Highlight best
best_idx = thresholds_plot.index(best_overall)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

fig.suptitle(f'Extinction Threshold Parameter Sweep (N={N_SAMPLES} samples)\n' +
             f'Optimal: θ = {best_overall:.4f}',
             fontsize=13, fontweight='bold')

output_png = 'final figures/fig_threshold_sweep.png'
output_pdf = 'final figures/fig_threshold_sweep.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\nFigure saved:")
print(f"  {output_png}")
print(f"  {output_pdf}")

plt.close()

# ============================================================================
# Save results to file
# ============================================================================
output_txt = 'THRESHOLD_SWEEP_RESULTS.txt'
with open(output_txt, 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXTINCTION THRESHOLD PARAMETER SWEEP RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Number of samples: {N_SAMPLES}\n")
    f.write(f"Model: {MODEL_PATH}\n\n")

    f.write("BASELINE (No Correction):\n")
    f.write(f"  Mean R²: {baseline_r2:.4f}\n")
    f.write(f"  Std R²:  {results[0.0]['r2_std']:.4f}\n\n")

    f.write("ALL THRESHOLDS:\n")
    f.write(f"{'Threshold':<12} {'Mean R²':<10} {'Δ vs Base':<12} {'% > 0.90':<10} {'% < 0.70':<10} {'Extinctions':<12}\n")
    f.write("-" * 80 + "\n")

    for threshold in thresholds_to_test:
        r = results[threshold]
        delta = r['r2_mean'] - baseline_r2
        f.write(f"{threshold:<12.4f} {r['r2_mean']:<10.4f} {delta:+<12.4f} "
                f"{r['pct_above_90']:<10.1f} {r['pct_below_70']:<10.1f} "
                f"{r['extinctions_applied']:<12d}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("RECOMMENDATIONS:\n")
    f.write("="*80 + "\n\n")
    f.write(f"Best for Mean R²:        θ = {best_threshold_mean:.4f} (R² = {best_r2_mean:.4f})\n")
    f.write(f"Best for % High Quality: θ = {best_threshold_quality:.4f} (% > 0.90 = {best_pct_90:.1f}%)\n")
    f.write(f"Recommended Overall:     θ = {best_overall:.4f}\n\n")

    f.write("INTERPRETATION:\n")
    f.write(f"  The optimal threshold balances:\n")
    f.write(f"    - Maximizing mean R²\n")
    f.write(f"    - Maximizing % high-quality samples (R² > 0.90)\n")
    f.write(f"    - Minimizing failures (R² < 0.70)\n")
    f.write(f"    - Applying corrections only when truly needed\n\n")

    f.write(f"  Threshold = {best_overall:.4f} provides the best balance.\n")
    f.write(f"  It improves mean R² by {(results[best_overall]['r2_mean'] - baseline_r2)*100:+.2f}%\n")
    f.write(f"  and affects {results[best_overall]['extinctions_applied']}/{N_SAMPLES} samples.\n")

print(f"\nResults saved to: {output_txt}")

print("\n" + "="*80)
print("✓ Threshold sweep complete!")
print("="*80)
print(f"\nRECOMMENDED THRESHOLD: {best_overall:.4f}")
print("="*80)
