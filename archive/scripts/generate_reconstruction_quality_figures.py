"""
Generate publication-quality reconstruction quality figures for Chaos, Solitons and Fractals.

This script creates comprehensive visualizations analyzing the CVAE's reconstruction performance:
1. Qualitative reconstruction examples with stochastic uncertainty
2. Quantitative metrics (MSE, MAE, R²) across all test samples
3. Per-curve reconstruction error analysis
4. Max value prediction accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

# Color palette for professional look
COLORS = sns.color_palette("husl", 7)
ERROR_COLOR = '#e74c3c'
PRED_COLOR = '#3498db'

# Import model and config
from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE
from utils.post_processing import apply_extinction_threshold

# Model configuration matching the trained model
model_config = {
    "in_channels": 7,
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 50,  # Match the trained model
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,  # NEW: Scale-conditioned model
}

# Data paths
TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
MODEL_PATH = 'model_ckpts/model_final_50_conditioned.pth'  # NEW: Conditioned model
OUTPUT_DIR = 'final figures/'

# Number of stochastic samples for uncertainty quantification
N_STOCHASTIC_SAMPLES = 50


def load_model_and_data():
    """Load the trained model and test dataset."""
    print("Loading model and data...")

    # Load model
    model = LSTM_VAE(config=model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load test data
    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)

    test_data = data_package['data']  # Normalized curves
    test_max_vals = data_package['reconstruction_max_values']  # Original scale max values

    print(f"Model loaded. Test samples: {len(test_data)}")
    return model, test_data, test_max_vals


def compute_reconstruction_metrics(model, test_data, test_max_vals):
    """
    Compute comprehensive reconstruction metrics across the test set.

    Returns:
        dict: Contains various reconstruction quality metrics
    """
    print("Computing reconstruction metrics on test set...")

    n_samples = len(test_data)
    n_curves = test_data[0].shape[0]
    seq_len = test_data[0].shape[1]

    # Storage for predictions and ground truth
    all_recons_norm = np.zeros((n_samples, n_curves, seq_len))
    all_original_norm = np.zeros((n_samples, n_curves, seq_len))
    all_recons_denorm = np.zeros((n_samples, n_curves, seq_len))
    all_original_denorm = np.zeros((n_samples, n_curves, seq_len))
    all_max_vals_pred = np.zeros((n_samples, n_curves))
    all_max_vals_true = np.zeros((n_samples, n_curves))

    # For stochastic uncertainty (on subset)
    stochastic_uncertainty = []

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Processing test samples"):
            # Prepare input
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals_true = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Single forward pass (with max_vals for conditioning)
            recon_norm, _, _, max_vals_pred, _ = model(x_norm, max_vals_true, teacher_forcing_ratio=0.0)

            # Convert to numpy
            x_norm_np = x_norm.cpu().numpy()[0]
            recon_norm_np = recon_norm.cpu().numpy()[0]
            max_vals_pred_np = max_vals_pred.cpu().numpy()[0]
            max_vals_true_np = max_vals_true.cpu().numpy()[0]  # Remove batch dimension

            # Denormalize using predicted max values (what model actually produces)
            x_denorm_np = x_norm_np * max_vals_true_np[:, np.newaxis]
            recon_denorm_np = recon_norm_np * max_vals_pred_np[:, np.newaxis]

            # Apply extinction threshold fix to reconstruction (optimal threshold from sweep)
            recon_denorm_np = apply_extinction_threshold(
                recon_denorm_np.T, threshold=0.005, extinction_value=0.0005
            ).T

            # Store
            all_original_norm[i] = x_norm_np
            all_recons_norm[i] = recon_norm_np
            all_original_denorm[i] = x_denorm_np
            all_recons_denorm[i] = recon_denorm_np
            all_max_vals_pred[i] = max_vals_pred_np
            all_max_vals_true[i] = max_vals_true_np

            # Compute stochastic uncertainty for subset of samples
            if i % 50 == 0:  # Every 50th sample
                stochastic_samples = []
                for _ in range(N_STOCHASTIC_SAMPLES):
                    recon_s, _, _, max_pred_s, _ = model(x_norm, max_vals_true, teacher_forcing_ratio=0.0)
                    # Denormalize and apply extinction fix
                    recon_s_denorm = recon_s.cpu().numpy()[0] * max_pred_s.cpu().numpy()[0][:, np.newaxis]
                    recon_s_fixed = apply_extinction_threshold(
                        recon_s_denorm.T, threshold=0.005, extinction_value=0.0005
                    ).T
                    stochastic_samples.append(recon_s_fixed)
                stochastic_samples = np.array(stochastic_samples)  # (N_SAMPLES, n_curves, seq_len)
                uncertainty = np.std(stochastic_samples, axis=0)  # (n_curves, seq_len)
                stochastic_uncertainty.append(uncertainty.mean())  # Mean uncertainty across curves and time

    # Compute metrics
    metrics = {
        'all_original_norm': all_original_norm,
        'all_recons_norm': all_recons_norm,
        'all_original_denorm': all_original_denorm,
        'all_recons_denorm': all_recons_denorm,
        'all_max_vals_pred': all_max_vals_pred,
        'all_max_vals_true': all_max_vals_true,
    }

    # MSE, MAE, RMSE per curve (normalized)
    mse_per_curve_norm = np.mean((all_original_norm - all_recons_norm)**2, axis=(0, 2))
    mae_per_curve_norm = np.mean(np.abs(all_original_norm - all_recons_norm), axis=(0, 2))
    rmse_per_curve_norm = np.sqrt(mse_per_curve_norm)

    # MSE, MAE, RMSE per curve (denormalized)
    mse_per_curve_denorm = np.mean((all_original_denorm - all_recons_denorm)**2, axis=(0, 2))
    mae_per_curve_denorm = np.mean(np.abs(all_original_denorm - all_recons_denorm), axis=(0, 2))
    rmse_per_curve_denorm = np.sqrt(mse_per_curve_denorm)

    # R² per curve (normalized)
    r2_per_curve_norm = []
    for c in range(n_curves):
        y_true = all_original_norm[:, c, :].flatten()
        y_pred = all_recons_norm[:, c, :].flatten()
        r2 = r2_score(y_true, y_pred)
        r2_per_curve_norm.append(r2)

    # R² per curve (denormalized)
    r2_per_curve_denorm = []
    for c in range(n_curves):
        y_true = all_original_denorm[:, c, :].flatten()
        y_pred = all_recons_denorm[:, c, :].flatten()
        r2 = r2_score(y_true, y_pred)
        r2_per_curve_denorm.append(r2)

    # Max value prediction metrics
    max_val_mse = np.mean((all_max_vals_true - all_max_vals_pred)**2, axis=0)
    max_val_mae = np.mean(np.abs(all_max_vals_true - all_max_vals_pred), axis=0)
    max_val_r2 = []
    for c in range(n_curves):
        if c == 0:  # Skip curve 0 (always 1.0)
            max_val_r2.append(1.0)
        else:
            r2 = r2_score(all_max_vals_true[:, c], all_max_vals_pred[:, c])
            max_val_r2.append(r2)

    metrics.update({
        'mse_per_curve_norm': mse_per_curve_norm,
        'mae_per_curve_norm': mae_per_curve_norm,
        'rmse_per_curve_norm': rmse_per_curve_norm,
        'r2_per_curve_norm': r2_per_curve_norm,
        'mse_per_curve_denorm': mse_per_curve_denorm,
        'mae_per_curve_denorm': mae_per_curve_denorm,
        'rmse_per_curve_denorm': rmse_per_curve_denorm,
        'r2_per_curve_denorm': r2_per_curve_denorm,
        'max_val_mse': max_val_mse,
        'max_val_mae': max_val_mae,
        'max_val_r2': max_val_r2,
        'mean_stochastic_uncertainty': np.mean(stochastic_uncertainty),
    })

    return metrics


def generate_qualitative_examples(model, test_data, test_max_vals):
    """
    Generate Figure 1: Qualitative reconstruction examples with stochastic uncertainty.
    """
    print("Generating qualitative reconstruction examples...")

    # Select diverse examples
    indices = [10, 150, 300, 450]  # Different test samples
    n_examples = len(indices)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, n_examples, figure=fig, hspace=0.35, wspace=0.3)

    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(indices):
            # Prepare data
            x_norm = torch.tensor(test_data[sample_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals_true = torch.tensor(test_max_vals[sample_idx], dtype=torch.float32)

            # Generate multiple stochastic samples (with max_vals for conditioning)
            recons_norm_samples = []
            max_vals_pred_samples = []
            max_vals_true_tensor = torch.tensor(test_max_vals[sample_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            for _ in range(N_STOCHASTIC_SAMPLES):
                recon, _, _, max_pred, _ = model(x_norm, max_vals_true_tensor, teacher_forcing_ratio=0.0)
                recons_norm_samples.append(recon.cpu().numpy()[0])
                max_vals_pred_samples.append(max_pred.cpu().numpy()[0])

            recons_norm_samples = np.array(recons_norm_samples)  # (N_SAMPLES, n_curves, seq_len)
            max_vals_pred_samples = np.array(max_vals_pred_samples)  # (N_SAMPLES, n_curves)

            # Compute mean and std
            recon_mean = recons_norm_samples.mean(axis=0)
            recon_std = recons_norm_samples.std(axis=0)
            max_pred_mean = max_vals_pred_samples.mean(axis=0)

            # Denormalize
            x_denorm = test_data[sample_idx] * test_max_vals[sample_idx][:, np.newaxis]

            # Denormalize all stochastic samples and apply extinction fix
            recons_denorm_samples = []
            for i in range(len(recons_norm_samples)):
                recon_denorm = recons_norm_samples[i] * max_vals_pred_samples[i][:, np.newaxis]
                # Apply extinction threshold fix (optimal threshold from sweep)
                recon_denorm_fixed = apply_extinction_threshold(
                    recon_denorm.T, threshold=0.005, extinction_value=0.0005
                ).T
                recons_denorm_samples.append(recon_denorm_fixed)

            recons_denorm_samples = np.array(recons_denorm_samples)
            recon_denorm_mean = recons_denorm_samples.mean(axis=0)
            recon_denorm_std = recons_denorm_samples.std(axis=0)

            # Plot normalized
            ax_norm = fig.add_subplot(gs[0, plot_idx])
            time_steps = np.arange(test_data[sample_idx].shape[1])
            for c in range(7):
                ax_norm.plot(time_steps, test_data[sample_idx][c],
                           color=COLORS[c], linewidth=1.5, alpha=0.8, label=f'$x_{c}$')
                ax_norm.plot(time_steps, recon_mean[c],
                           color=COLORS[c], linewidth=1.2, linestyle='--', alpha=0.7)
                ax_norm.fill_between(time_steps,
                                    recon_mean[c] - 2*recon_std[c],
                                    recon_mean[c] + 2*recon_std[c],
                                    color=COLORS[c], alpha=0.15)

            ax_norm.set_xlabel('Time step')
            ax_norm.set_ylabel('Normalized value')
            ax_norm.set_title(f'Sample {sample_idx} (Normalized)', fontweight='bold')
            ax_norm.grid(True, alpha=0.3, linewidth=0.5)
            ax_norm.set_ylim([-0.05, 1.1])
            if plot_idx == 0:
                ax_norm.legend(loc='upper right', ncol=2, framealpha=0.9)

            # Plot denormalized
            ax_denorm = fig.add_subplot(gs[1, plot_idx])
            for c in range(7):
                ax_denorm.plot(time_steps, x_denorm[c],
                             color=COLORS[c], linewidth=1.5, alpha=0.8)
                ax_denorm.plot(time_steps, recon_denorm_mean[c],
                             color=COLORS[c], linewidth=1.2, linestyle='--', alpha=0.7)
                ax_denorm.fill_between(time_steps,
                                      recon_denorm_mean[c] - 2*recon_denorm_std[c],
                                      recon_denorm_mean[c] + 2*recon_denorm_std[c],
                                      color=COLORS[c], alpha=0.15)

            ax_denorm.set_xlabel('Time step')
            ax_denorm.set_ylabel('Population density')
            ax_denorm.set_title(f'Sample {sample_idx} (Original Scale)', fontweight='bold')
            ax_denorm.grid(True, alpha=0.3, linewidth=0.5)

    # Add legend explaining line styles
    fig.text(0.5, 0.97, 'Solid lines: Original | Dashed lines: Reconstruction (mean) | Shaded: 95% CI',
             ha='center', fontsize=10, style='italic')

    plt.savefig(f'{OUTPUT_DIR}fig_reconstruction_examples.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}fig_reconstruction_examples.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}fig_reconstruction_examples.png/pdf")
    plt.close()


def generate_quantitative_metrics_figure(metrics):
    """
    Generate Figure 2: Quantitative reconstruction metrics per curve.
    """
    print("Generating quantitative metrics figure...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    curve_labels = [f'$x_{i}$' for i in range(7)]
    x_pos = np.arange(7)

    # Row 1: Normalized metrics
    # MSE
    ax = axes[0, 0]
    bars = ax.bar(x_pos, metrics['mse_per_curve_norm'], color=COLORS, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Curve index')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error (Normalized)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(curve_labels)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # MAE
    ax = axes[0, 1]
    bars = ax.bar(x_pos, metrics['mae_per_curve_norm'], color=COLORS, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Curve index')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error (Normalized)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(curve_labels)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # R²
    ax = axes[0, 2]
    bars = ax.bar(x_pos, metrics['r2_per_curve_norm'], color=COLORS, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Curve index')
    ax.set_ylabel('R²')
    ax.set_title('Coefficient of Determination (Normalized)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(curve_labels)
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.7, label='0.95 threshold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.legend()
    ax.set_ylim([0.85, 1.0])

    # Row 2: Denormalized metrics
    # MSE
    ax = axes[1, 0]
    bars = ax.bar(x_pos, metrics['mse_per_curve_denorm'], color=COLORS, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Curve index')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error (Original Scale)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(curve_labels)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # MAE
    ax = axes[1, 1]
    bars = ax.bar(x_pos, metrics['mae_per_curve_denorm'], color=COLORS, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Curve index')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error (Original Scale)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(curve_labels)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # R²
    ax = axes[1, 2]
    bars = ax.bar(x_pos, metrics['r2_per_curve_denorm'], color=COLORS, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Curve index')
    ax.set_ylabel('R²')
    ax.set_title('Coefficient of Determination (Original Scale)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(curve_labels)
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.7, label='0.95 threshold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.legend()
    ax.set_ylim([0.85, 1.0])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_reconstruction_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}fig_reconstruction_metrics.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}fig_reconstruction_metrics.png/pdf")
    plt.close()


def generate_error_analysis_figure(metrics):
    """
    Generate Figure 3: Reconstruction error distribution and analysis.
    """
    print("Generating error analysis figure...")

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # 1. Error distribution per curve (normalized)
    ax1 = fig.add_subplot(gs[0, :2])
    errors_norm = metrics['all_original_norm'] - metrics['all_recons_norm']  # (n_samples, n_curves, seq_len)

    positions = []
    data_to_plot = []
    labels = []
    for c in range(7):
        errors_flat = errors_norm[:, c, :].flatten()
        data_to_plot.append(errors_flat)
        positions.append(c)
        labels.append(f'$x_{c}$')

    bp = ax1.boxplot(data_to_plot, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(linewidth=0.8),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8),
                     medianprops=dict(linewidth=1.2, color='red'))

    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Curve index')
    ax1.set_ylabel('Reconstruction error')
    ax1.set_title('Reconstruction Error Distribution (Normalized)', fontweight='bold')
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)

    # 2. Temporal error pattern (averaged across all samples and curves)
    ax2 = fig.add_subplot(gs[0, 2])
    temporal_mae = np.mean(np.abs(errors_norm), axis=(0, 1))  # Average across samples and curves
    temporal_std = np.std(np.abs(errors_norm), axis=(0, 1))
    time_steps = np.arange(len(temporal_mae))

    ax2.plot(time_steps, temporal_mae, color=ERROR_COLOR, linewidth=1.5)
    ax2.fill_between(time_steps,
                     temporal_mae - temporal_std,
                     temporal_mae + temporal_std,
                     color=ERROR_COLOR, alpha=0.3)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Mean absolute error')
    ax2.set_title('Temporal Error Pattern', fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    # 3. Error correlation matrix between curves
    ax3 = fig.add_subplot(gs[1, 0])
    error_per_curve_per_sample = np.mean(np.abs(errors_norm), axis=2)  # (n_samples, n_curves)
    corr_matrix = np.corrcoef(error_per_curve_per_sample.T)

    im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xticks(np.arange(7))
    ax3.set_yticks(np.arange(7))
    ax3.set_xticklabels([f'$x_{i}$' for i in range(7)])
    ax3.set_yticklabels([f'$x_{i}$' for i in range(7)])
    ax3.set_title('Error Correlation Between Curves', fontweight='bold')

    # Add correlation values
    for i in range(7):
        for j in range(7):
            text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                          fontsize=8)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Correlation coefficient')

    # 4. Scatter plot: Original vs Reconstructed (all curves combined, normalized)
    ax4 = fig.add_subplot(gs[1, 1])
    # Sample subset for visualization
    sample_indices = np.random.choice(len(metrics['all_original_norm']), size=500, replace=False)
    for c in range(7):
        orig = metrics['all_original_norm'][sample_indices, c, :].flatten()
        recon = metrics['all_recons_norm'][sample_indices, c, :].flatten()
        ax4.scatter(orig, recon, alpha=0.3, s=1, color=COLORS[c], label=f'$x_{c}$')

    # Perfect reconstruction line
    lim_min = 0
    lim_max = 1.05
    ax4.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, alpha=0.5, label='Perfect reconstruction')
    ax4.set_xlabel('Original value')
    ax4.set_ylabel('Reconstructed value')
    ax4.set_title('Original vs Reconstructed (Normalized)', fontweight='bold')
    ax4.legend(loc='lower right', markerscale=5, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    ax4.set_xlim([lim_min, lim_max])
    ax4.set_ylim([lim_min, lim_max])
    ax4.set_aspect('equal')

    # 5. RMSE comparison (normalized vs denormalized)
    ax5 = fig.add_subplot(gs[1, 2])
    x_pos = np.arange(7)
    width = 0.35

    ax5.bar(x_pos - width/2, metrics['rmse_per_curve_norm'], width,
           label='Normalized', color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8)
    ax5.bar(x_pos + width/2, metrics['rmse_per_curve_denorm']/np.max(metrics['all_original_denorm'], axis=(0,2)),
           width, label='Original scale (normalized by max)', color='coral', alpha=0.8, edgecolor='black', linewidth=0.8)

    ax5.set_xlabel('Curve index')
    ax5.set_ylabel('RMSE')
    ax5.set_title('RMSE Comparison', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'$x_{i}$' for i in range(7)])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.savefig(f'{OUTPUT_DIR}fig_reconstruction_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}fig_reconstruction_error_analysis.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}fig_reconstruction_error_analysis.png/pdf")
    plt.close()


def generate_max_value_prediction_figure(metrics):
    """
    Generate Figure 4: Max value prediction accuracy analysis.
    """
    print("Generating max value prediction figure...")

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.4)

    # 1. Scatter plots for each curve's max value prediction
    for c in range(1, 7):  # Skip curve 0 (always 1.0)
        row = (c - 1) // 3
        col = (c - 1) % 3
        ax = fig.add_subplot(gs[row, col])

        true_vals = metrics['all_max_vals_true'][:, c]
        pred_vals = metrics['all_max_vals_pred'][:, c]

        ax.scatter(true_vals, pred_vals, alpha=0.5, s=10, color=COLORS[c])

        # Perfect prediction line
        val_min = min(true_vals.min(), pred_vals.min())
        val_max = max(true_vals.max(), pred_vals.max())
        ax.plot([val_min, val_max], [val_min, val_max], 'k--', linewidth=1, alpha=0.5)

        # Compute R² and display
        r2 = metrics['max_val_r2'][c]
        mae = metrics['max_val_mae'][c]

        ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}\nMAE = {mae:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)

        ax.set_xlabel(f'True max($x_{c}$)')
        ax.set_ylabel(f'Predicted max($x_{c}$)')
        ax.set_title(f'Curve {c} Max Value Prediction', fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')

    # 2. Overall metrics comparison
    ax = fig.add_subplot(gs[:, 3])

    x_pos = np.arange(1, 7)
    width = 0.25

    # R² scores
    r2_vals = [metrics['max_val_r2'][c] for c in range(1, 7)]
    ax.bar(x_pos - width, r2_vals, width, label='$R^2$',
          color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8)

    # Normalized MAE (divide by mean to make comparable)
    mae_normalized = []
    for c in range(1, 7):
        mean_val = metrics['all_max_vals_true'][:, c].mean()
        mae_norm = metrics['max_val_mae'][c] / mean_val
        mae_normalized.append(mae_norm)

    ax.bar(x_pos, mae_normalized, width, label='Normalized MAE',
          color='coral', alpha=0.8, edgecolor='black', linewidth=0.8)

    # Normalized RMSE
    rmse_normalized = []
    for c in range(1, 7):
        mean_val = metrics['all_max_vals_true'][:, c].mean()
        rmse = np.sqrt(metrics['max_val_mse'][c])
        rmse_norm = rmse / mean_val
        rmse_normalized.append(rmse_norm)

    ax.bar(x_pos + width, rmse_normalized, width, label='Normalized RMSE',
          color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Curve index')
    ax.set_ylabel('Metric value')
    ax.set_title('Max Value Prediction Metrics\n(Curves 1-6)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$x_{i}$' for i in range(1, 7)])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.savefig(f'{OUTPUT_DIR}fig_max_value_prediction.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}fig_max_value_prediction.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}fig_max_value_prediction.png/pdf")
    plt.close()


def print_summary_statistics(metrics):
    """Print a summary of key reconstruction quality metrics."""
    print("\n" + "="*80)
    print("RECONSTRUCTION QUALITY SUMMARY")
    print("="*80)

    print("\n--- Normalized Reconstruction Metrics ---")
    print(f"Overall MSE:  {np.mean(metrics['mse_per_curve_norm']):.6f}")
    print(f"Overall MAE:  {np.mean(metrics['mae_per_curve_norm']):.6f}")
    print(f"Overall RMSE: {np.mean(metrics['rmse_per_curve_norm']):.6f}")
    print(f"Overall R²:   {np.mean(metrics['r2_per_curve_norm']):.6f}")

    print("\n--- Original Scale Reconstruction Metrics ---")
    print(f"Overall MSE:  {np.mean(metrics['mse_per_curve_denorm']):.6f}")
    print(f"Overall MAE:  {np.mean(metrics['mae_per_curve_denorm']):.6f}")
    print(f"Overall RMSE: {np.mean(metrics['rmse_per_curve_denorm']):.6f}")
    print(f"Overall R²:   {np.mean(metrics['r2_per_curve_denorm']):.6f}")

    print("\n--- Max Value Prediction Metrics (Curves 1-6) ---")
    print(f"Overall MSE:  {np.mean(metrics['max_val_mse'][1:]):.6f}")
    print(f"Overall MAE:  {np.mean(metrics['max_val_mae'][1:]):.6f}")
    print(f"Overall R²:   {np.mean(metrics['max_val_r2'][1:]):.6f}")

    print("\n--- Per-Curve R² (Normalized) ---")
    for c in range(7):
        print(f"Curve {c}: {metrics['r2_per_curve_norm'][c]:.6f}")

    print("\n--- Per-Curve R² (Original Scale) ---")
    for c in range(7):
        print(f"Curve {c}: {metrics['r2_per_curve_denorm'][c]:.6f}")

    print("\n--- Stochastic Uncertainty ---")
    print(f"Mean uncertainty (std): {metrics['mean_stochastic_uncertainty']:.6f}")

    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("GENERATING RECONSTRUCTION QUALITY FIGURES FOR CHAOS, SOLITONS AND FRACTALS")
    print("="*80)

    # Load model and data
    model, test_data, test_max_vals = load_model_and_data()

    # Compute comprehensive metrics
    metrics = compute_reconstruction_metrics(model, test_data, test_max_vals)

    # Generate figures
    generate_qualitative_examples(model, test_data, test_max_vals)
    generate_quantitative_metrics_figure(metrics)
    generate_error_analysis_figure(metrics)
    generate_max_value_prediction_figure(metrics)

    # Print summary
    print_summary_statistics(metrics)

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"Figures saved in: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
