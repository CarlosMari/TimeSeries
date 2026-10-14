"""
Analyze failure cases: Where and why does the VAE generate samples with poor LV adherence?

Creates a figure showing:
1. Distribution of R² scores with worst cases highlighted
2. Example worst-case trajectories
3. Per-species R² breakdown for worst cases
4. Comparison to best cases
5. Analysis of what features cause low R²
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

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.models.cvae import LSTM_VAE

# Professional style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 11,
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

# Load model
print("Loading model...")
model = LSTM_VAE(model_config).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded!\n")

def test_lv_fit(x_data):
    """Test LV adherence - returns per-species R² scores."""
    N = x_data.shape[1]
    r2_scores = []
    growth_rates_actual = []
    growth_rates_predicted = []

    for i in range(N):
        x_i_positive = np.maximum(x_data[:, i], 1e-9)
        log_xi = np.log(x_i_positive)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d_log_xi_dt = np.gradient(log_xi)

        # Fit: d(log(x_i))/dt = r_i + Σ_j a_ij * x_j
        model_lr = LinearRegression()
        model_lr.fit(x_data, d_log_xi_dt)
        predicted = model_lr.predict(x_data)

        r2 = r2_score(d_log_xi_dt, predicted)
        r2_scores.append(r2)
        growth_rates_actual.append(d_log_xi_dt)
        growth_rates_predicted.append(predicted)

    return np.array(r2_scores), growth_rates_actual, growth_rates_predicted

def analyze_sample_features(time_series):
    """Extract features that might correlate with poor R²."""
    features = {}

    # Total variance
    features['total_variance'] = np.sum(np.var(time_series, axis=0))

    # Mean abundance
    features['mean_abundance'] = np.mean(time_series)

    # Number of species near zero
    features['species_near_zero'] = np.sum(np.min(time_series, axis=0) < 0.01)

    # Maximum abundance
    features['max_abundance'] = np.max(time_series)

    # Coefficient of variation
    features['cv'] = np.std(time_series) / (np.mean(time_series) + 1e-9)

    # Trend strength (how much does it deviate from flat)
    trends = []
    for i in range(time_series.shape[1]):
        trend = np.polyfit(np.arange(len(time_series)), time_series[:, i], 1)[0]
        trends.append(abs(trend))
    features['mean_abs_trend'] = np.mean(trends)

    # Oscillation count (peaks + troughs)
    from scipy import signal
    extrema_counts = []
    for i in range(time_series.shape[1]):
        peaks, _ = signal.find_peaks(time_series[:, i])
        troughs, _ = signal.find_peaks(-time_series[:, i])
        extrema_counts.append(len(peaks) + len(troughs))
    features['mean_extrema'] = np.mean(extrema_counts)

    return features

# ============================================================================
# Generate samples and compute R²
# ============================================================================
print("Generating samples and computing R² scores...")
N_SAMPLES = 1000
samples_data = []
r2_scores_mean = []
r2_scores_per_species = []
sample_features = []

with torch.no_grad():
    for i in range(N_SAMPLES):
        if (i + 1) % 200 == 0:
            print(f"  Generated {i+1}/{N_SAMPLES}...")

        z = torch.randn(1, 30).to(device)
        recon_norm, max_vals_pred = model.decode(z)

        recon_norm_np = recon_norm.cpu().numpy()[0]
        max_vals_np = max_vals_pred.cpu().numpy()[0]

        time_series = recon_norm_np * max_vals_np.reshape(-1, 1)
        time_series = time_series.T  # (seq_len, n_species)

        r2_per_species, growth_actual, growth_pred = test_lv_fit(time_series)
        features = analyze_sample_features(time_series)

        samples_data.append({
            'time_series': time_series,
            'r2_per_species': r2_per_species,
            'growth_actual': growth_actual,
            'growth_pred': growth_pred,
            'features': features
        })

        r2_scores_mean.append(np.mean(r2_per_species))
        r2_scores_per_species.append(r2_per_species)
        sample_features.append(features)

r2_scores_mean = np.array(r2_scores_mean)
r2_scores_per_species = np.array(r2_scores_per_species)

print(f"\nR² Statistics:")
print(f"  Mean: {np.mean(r2_scores_mean):.4f}")
print(f"  Median: {np.median(r2_scores_mean):.4f}")
print(f"  Std: {np.std(r2_scores_mean):.4f}")
print(f"  Min: {np.min(r2_scores_mean):.4f}")
print(f"  Max: {np.max(r2_scores_mean):.4f}")

# ============================================================================
# Find worst and best cases
# ============================================================================
worst_indices = np.argsort(r2_scores_mean)[:5]  # 5 worst
best_indices = np.argsort(r2_scores_mean)[-5:]  # 5 best

print(f"\nWorst 5 R² scores: {r2_scores_mean[worst_indices]}")
print(f"Best 5 R² scores: {r2_scores_mean[best_indices]}")

# ============================================================================
# Analyze what differentiates worst from best
# ============================================================================
print("\n" + "="*80)
print("FEATURE ANALYSIS: What makes samples have low R²?")
print("="*80)

feature_names = list(sample_features[0].keys())
for feat_name in feature_names:
    worst_vals = [sample_features[i][feat_name] for i in worst_indices]
    best_vals = [sample_features[i][feat_name] for i in best_indices]

    print(f"\n{feat_name}:")
    print(f"  Worst cases: {np.mean(worst_vals):.4f} ± {np.std(worst_vals):.4f}")
    print(f"  Best cases:  {np.mean(best_vals):.4f} ± {np.std(best_vals):.4f}")
    print(f"  Difference:  {np.mean(worst_vals) - np.mean(best_vals):+.4f}")

# ============================================================================
# Create comprehensive failure analysis figure
# ============================================================================
print("\nCreating failure analysis figure...")

fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.4,
                       left=0.06, right=0.98, top=0.93, bottom=0.06)

colors = plt.cm.tab10(np.linspace(0, 0.7, 7))

# ============================================================================
# Panel A: R² Distribution with worst cases highlighted
# ============================================================================
ax_hist = fig.add_subplot(gs[0, :3])

bins = np.linspace(r2_scores_mean.min(), 1.0, 50)
ax_hist.hist(r2_scores_mean, bins=bins, alpha=0.7, color='steelblue',
             edgecolor='black', linewidth=0.5)

# Highlight worst cases
for idx in worst_indices:
    ax_hist.axvline(r2_scores_mean[idx], color='red', linestyle='--',
                    linewidth=1.5, alpha=0.7)

ax_hist.axvline(np.median(r2_scores_mean), color='green', linestyle='-',
                linewidth=2.5, alpha=0.8, label=f'Median = {np.median(r2_scores_mean):.3f}')

ax_hist.set_xlabel('Mean R² (LV Adherence)', fontweight='bold')
ax_hist.set_ylabel('Frequency', fontweight='bold')
ax_hist.set_title('A) Distribution of R² Scores (Worst Cases in Red)',
                  fontweight='bold', loc='left', pad=10)
ax_hist.legend(loc='upper left', frameon=True)
ax_hist.grid(True, alpha=0.3)

# ============================================================================
# Panel B: Per-species R² breakdown for worst cases
# ============================================================================
ax_breakdown = fig.add_subplot(gs[0, 3:])

worst_r2_matrix = r2_scores_per_species[worst_indices]  # (5, 7)
species_labels = [f'Sp{i+1}' for i in range(7)]

x_pos = np.arange(7)
width = 0.15

for i, idx in enumerate(worst_indices):
    offset = (i - 2) * width
    ax_breakdown.bar(x_pos + offset, worst_r2_matrix[i], width,
                     label=f'Sample {i+1} (R²={r2_scores_mean[idx]:.3f})',
                     alpha=0.8)

ax_breakdown.set_xlabel('Species', fontweight='bold')
ax_breakdown.set_ylabel('R² per Species', fontweight='bold')
ax_breakdown.set_title('B) Per-Species R² for Worst Cases',
                       fontweight='bold', loc='left', pad=10)
ax_breakdown.set_xticks(x_pos)
ax_breakdown.set_xticklabels(species_labels)
ax_breakdown.axhline(0.9, color='green', linestyle='--', linewidth=1.5,
                     alpha=0.5, label='Good threshold (0.9)')
ax_breakdown.legend(loc='lower right', frameon=True, fontsize=6, ncol=2)
ax_breakdown.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Rows 2-3: Worst case examples (show 3 worst)
# ============================================================================
time = np.arange(65)

for plot_idx in range(3):
    sample_idx = worst_indices[plot_idx]
    sample = samples_data[sample_idx]

    # Trajectory
    ax_traj = fig.add_subplot(gs[1 + plot_idx // 2, (plot_idx % 2) * 2:(plot_idx % 2) * 2 + 2])

    for i in range(7):
        ax_traj.plot(time, sample['time_series'][:, i], color=colors[i],
                    linewidth=1.5, alpha=0.8, label=f'Sp{i+1}')

    ax_traj.set_xlabel('Time', fontweight='bold')
    ax_traj.set_ylabel('Abundance', fontweight='bold')
    ax_traj.set_title(f'{chr(67+plot_idx)}) Worst Case {plot_idx+1} (R² = {r2_scores_mean[sample_idx]:.3f})',
                     fontweight='bold', loc='left', pad=8)
    ax_traj.legend(loc='upper right', frameon=True, fontsize=6, ncol=2)
    ax_traj.grid(True, alpha=0.3)

    # Per-species LV fit
    ax_fit = fig.add_subplot(gs[1 + plot_idx // 2, (plot_idx % 2) * 2 + 2:(plot_idx % 2) * 2 + 3])

    species_x = np.arange(7)
    species_r2 = sample['r2_per_species']

    bars = ax_fit.bar(species_x, species_r2, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Color bars by quality
    for i, bar in enumerate(bars):
        if species_r2[i] < 0.7:
            bar.set_color('red')
        elif species_r2[i] < 0.9:
            bar.set_color('orange')

    ax_fit.axhline(0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_fit.set_xlabel('Species', fontweight='bold', fontsize=8)
    ax_fit.set_ylabel('R²', fontweight='bold', fontsize=8)
    ax_fit.set_title(f'Per-Species R²', fontweight='bold', fontsize=8, loc='left')
    ax_fit.set_xticks(species_x)
    ax_fit.set_xticklabels([f'{i+1}' for i in range(7)])
    ax_fit.set_ylim(-0.1, 1.05)
    ax_fit.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Row 4: Feature comparison - worst vs best
# ============================================================================
ax_features = fig.add_subplot(gs[3, :3])

feature_comparison = {}
for feat_name in feature_names:
    worst_vals = [sample_features[i][feat_name] for i in worst_indices]
    best_vals = [sample_features[i][feat_name] for i in best_indices]
    feature_comparison[feat_name] = {
        'worst_mean': np.mean(worst_vals),
        'best_mean': np.mean(best_vals),
        'worst_std': np.std(worst_vals),
        'best_std': np.std(best_vals)
    }

feature_short_names = {
    'total_variance': 'Tot. Var.',
    'mean_abundance': 'Mean Abd.',
    'species_near_zero': 'Near-Zero Sp.',
    'max_abundance': 'Max Abd.',
    'cv': 'CV',
    'mean_abs_trend': 'Trend',
    'mean_extrema': 'Extrema'
}

x_pos = np.arange(len(feature_names))
width = 0.35

worst_means = [feature_comparison[f]['worst_mean'] for f in feature_names]
best_means = [feature_comparison[f]['best_mean'] for f in feature_names]
worst_stds = [feature_comparison[f]['worst_std'] for f in feature_names]
best_stds = [feature_comparison[f]['best_std'] for f in feature_names]

# Normalize for visualization
worst_means_norm = (np.array(worst_means) - np.array(best_means)) / (np.array(best_means) + 1e-9)

ax_features.bar(x_pos, worst_means_norm, width, label='Worst cases',
               color='red', alpha=0.7, edgecolor='black', linewidth=1)

ax_features.axhline(0, color='black', linestyle='-', linewidth=1.5)
ax_features.set_xlabel('Feature', fontweight='bold')
ax_features.set_ylabel('Relative Difference\n(Worst - Best) / Best', fontweight='bold')
ax_features.set_title('F) Feature Differences: Worst vs Best Cases',
                     fontweight='bold', loc='left', pad=10)
ax_features.set_xticks(x_pos)
ax_features.set_xticklabels([feature_short_names[f] for f in feature_names], rotation=45, ha='right')
ax_features.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Panel G: Best case example (for comparison)
# ============================================================================
ax_best = fig.add_subplot(gs[3, 3:])

best_sample_idx = best_indices[-1]
best_sample = samples_data[best_sample_idx]

for i in range(7):
    ax_best.plot(time, best_sample['time_series'][:, i], color=colors[i],
                linewidth=1.5, alpha=0.8, label=f'Sp{i+1}')

ax_best.set_xlabel('Time', fontweight='bold')
ax_best.set_ylabel('Abundance', fontweight='bold')
ax_best.set_title(f'G) Best Case Example (R² = {r2_scores_mean[best_sample_idx]:.3f})',
                 fontweight='bold', loc='left', pad=8)
ax_best.legend(loc='upper right', frameon=True, fontsize=6, ncol=2)
ax_best.grid(True, alpha=0.3)

# Overall title
fig.suptitle('Failure Mode Analysis: Understanding Low R² Cases in VAE-Generated Samples',
             fontsize=12, fontweight='bold', y=0.97)

# Save
output_png = 'final figures/fig_failure_analysis.png'
output_pdf = 'final figures/fig_failure_analysis.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\nFigure saved:")
print(f"  {output_png}")
print(f"  {output_pdf}")

plt.close()

print("\n" + "="*80)
print("✓ Failure analysis complete!")
print("="*80)
