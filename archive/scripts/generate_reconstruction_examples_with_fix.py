"""
Generate reconstruction examples figure with extinction threshold fix.
Simplified version that only generates the qualitative examples figure.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
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
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

# Color palette
COLORS = sns.color_palette("husl", 7)

from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE
from utils.post_processing import apply_extinction_threshold

# Model configuration
model_config = {
    "in_channels": 7,
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 50,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
MODEL_PATH = 'model_ckpts/model_final_50_conditioned.pth'
OUTPUT_DIR = 'final figures/'
N_STOCHASTIC_SAMPLES = 50

print("="*80)
print("GENERATING RECONSTRUCTION EXAMPLES WITH EXTINCTION FIX")
print("="*80)

# Load model
print("Loading model...")
model = LSTM_VAE(config=model_config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load test data
print("Loading test data...")
with open(TEST_ROUTE, 'rb') as f:
    data_package = pickle.load(f)
test_data = data_package['data']
test_max_vals = data_package['reconstruction_max_values']

print(f"Model loaded. Test samples: {len(test_data)}")
print("\nGenerating qualitative reconstruction examples with extinction fix...")

# Select diverse examples
indices = [10, 150, 300, 450]
n_examples = len(indices)

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, n_examples, figure=fig, hspace=0.35, wspace=0.3)

with torch.no_grad():
    for plot_idx, sample_idx in enumerate(indices):
        print(f"  Processing sample {sample_idx}...")

        # Prepare data
        x_norm = torch.tensor(test_data[sample_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        max_vals_true = torch.tensor(test_max_vals[sample_idx], dtype=torch.float32)
        max_vals_true_tensor = torch.tensor(test_max_vals[sample_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Generate multiple stochastic samples
        recons_norm_samples = []
        max_vals_pred_samples = []

        for _ in range(N_STOCHASTIC_SAMPLES):
            recon, _, _, max_pred, _ = model(x_norm, max_vals_true_tensor, teacher_forcing_ratio=0.0)
            recons_norm_samples.append(recon.cpu().numpy()[0])
            max_vals_pred_samples.append(max_pred.cpu().numpy()[0])

        recons_norm_samples = np.array(recons_norm_samples)
        max_vals_pred_samples = np.array(max_vals_pred_samples)

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
            # Apply extinction threshold fix (optimal threshold from sweep: 0.005)
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

        # Plot denormalized with extinction fix
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

# Add legend
fig.text(0.5, 0.97, 'Solid lines: Original | Dashed lines: Reconstruction (mean) | Shaded: 95% CI | [Extinction fix applied: θ=0.005]',
         ha='center', fontsize=10, style='italic')

output_png = f'{OUTPUT_DIR}fig_reconstruction_examples.png'
output_pdf = f'{OUTPUT_DIR}fig_reconstruction_examples.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')

print(f"\n✓ Figure saved:")
print(f"  {output_png}")
print(f"  {output_pdf}")
print("\n" + "="*80)
print("DONE!")
print("="*80)
