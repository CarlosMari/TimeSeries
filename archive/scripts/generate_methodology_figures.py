import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the plotting style
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Figure 1: Model Architecture Diagram
def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    encoder_color = '#E8F4F8'
    bottleneck_color = '#FFE8E8'
    decoder_color = '#E8F8E8'
    predictor_color = '#FFF4E8'

    # Input
    input_box = FancyBboxPatch((0.5, 4.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 5.25, 'Input\nX', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.25, 4.1, '(N, 7, 65)', ha='center', va='center', fontsize=8)

    # Encoder LSTM
    encoder_box = FancyBboxPatch((2.5, 4), 2, 2.5, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=encoder_color, linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(3.5, 5.7, 'Bidirectional', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(3.5, 5.3, 'LSTM Encoder', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(3.5, 4.9, '256 hidden', ha='center', va='center', fontsize=9)
    ax.text(3.5, 4.6, '2 layers', ha='center', va='center', fontsize=9)
    ax.text(3.5, 4.3, 'Input: (N, 65, 7)', ha='center', va='center', fontsize=8, style='italic')

    # Arrow from input to encoder
    arrow1 = FancyArrowPatch((2, 5.25), (2.5, 5.25), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Bottleneck (mu, log_var, z)
    bottleneck_box = FancyBboxPatch((5, 3.5), 2, 3.5, boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor=bottleneck_color, linewidth=2)
    ax.add_patch(bottleneck_box)
    ax.text(6, 6.5, 'Bottleneck', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 6.0, 'FC → μ (30)', ha='center', va='center', fontsize=9)
    ax.text(6, 5.6, 'FC → log σ² (30)', ha='center', va='center', fontsize=9)
    ax.text(6, 5.0, 'Reparameterization:', ha='center', va='center', fontsize=9, style='italic')
    ax.text(6, 4.6, 'z = μ + ε·σ', ha='center', va='center', fontsize=9, style='italic')
    ax.text(6, 4.2, 'ε ~ N(0, I)', ha='center', va='center', fontsize=8, style='italic')
    ax.text(6, 3.8, 'z ∈ ℝ³⁰', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow from encoder to bottleneck
    arrow2 = FancyArrowPatch((4.5, 5.25), (5, 5.25), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)

    # Max Value Predictor (parallel branch from z)
    predictor_box = FancyBboxPatch((5, 0.5), 2, 2.5, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor=predictor_color, linewidth=2)
    ax.add_patch(predictor_box)
    ax.text(6, 2.5, 'Max Value', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 2.1, 'Predictor MLP', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 1.7, 'FC(30 → 15)', ha='center', va='center', fontsize=9)
    ax.text(6, 1.4, 'Dropout(0.2)', ha='center', va='center', fontsize=9)
    ax.text(6, 1.1, 'SiLU', ha='center', va='center', fontsize=9)
    ax.text(6, 0.8, 'FC(15 → 6)', ha='center', va='center', fontsize=9)

    # Arrow from z to predictor
    arrow_pred = FancyArrowPatch((6, 3.5), (6, 3.0), arrowstyle='->', mutation_scale=20, linewidth=2, color='darkorange')
    ax.add_patch(arrow_pred)
    ax.text(6.4, 3.25, 'z', ha='left', va='center', fontsize=9, color='darkorange', fontweight='bold')

    # Decoder LSTM
    decoder_box = FancyBboxPatch((7.5, 4), 2.5, 2.5, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=decoder_color, linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(8.75, 5.8, 'Autoregressive', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 5.4, 'LSTM Decoder', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 5.0, '256 hidden', ha='center', va='center', fontsize=9)
    ax.text(8.75, 4.7, '2 layers', ha='center', va='center', fontsize=9)
    ax.text(8.75, 4.3, 'Input: [x_t, z]', ha='center', va='center', fontsize=8, style='italic')

    # Arrow from bottleneck to decoder
    arrow3 = FancyArrowPatch((7, 5.25), (7.5, 5.25), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    ax.text(7.25, 5.6, 'z', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Output layer
    output_fc_box = FancyBboxPatch((10.5, 4.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(output_fc_box)
    ax.text(11.25, 5.5, 'Output FC', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(11.25, 5.1, '256 → 7', ha='center', va='center', fontsize=9)
    ax.text(11.25, 4.7, 'Clamp [0,1]', ha='center', va='center', fontsize=8, style='italic')

    # Arrow from decoder to output
    arrow4 = FancyArrowPatch((10, 5.25), (10.5, 5.25), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)

    # Final output
    final_output_box = FancyBboxPatch((12.5, 4.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                                     edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(final_output_box)
    ax.text(13.25, 5.25, 'X̂', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(13.25, 4.1, '(N, 7, 65)', ha='center', va='center', fontsize=8)

    # Arrow to final output
    arrow5 = FancyArrowPatch((12, 5.25), (12.5, 5.25), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)

    # Max values output
    max_output_box = FancyBboxPatch((7.5, 0.5), 1.5, 1, boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(max_output_box)
    ax.text(8.25, 1.0, 'Max Values', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8.25, 0.65, '(N, 7)', ha='center', va='center', fontsize=8)

    # Arrow from predictor to max output
    arrow_max = FancyArrowPatch((7, 1.5), (7.5, 1.2), arrowstyle='->', mutation_scale=20, linewidth=2, color='darkorange')
    ax.add_patch(arrow_max)

    # Title
    ax.text(7, 8.5, 'LSTM-VAE Architecture for GLV Reconstruction',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Teacher forcing indicator
    tf_box = FancyBboxPatch((8.75, 7.2), 3, 0.8, boxstyle="round,pad=0.05",
                           edgecolor='blue', facecolor='lightblue', linewidth=1.5, linestyle='--')
    ax.add_patch(tf_box)
    ax.text(10.25, 7.6, 'Teacher Forcing during Training', ha='center', va='center',
            fontsize=9, style='italic', color='darkblue')

    # Add legend for data flow
    legend_elements = [
        mpatches.Patch(facecolor=encoder_color, edgecolor='black', label='Encoder'),
        mpatches.Patch(facecolor=bottleneck_color, edgecolor='black', label='Latent Space'),
        mpatches.Patch(facecolor=decoder_color, edgecolor='black', label='Decoder'),
        mpatches.Patch(facecolor=predictor_color, edgecolor='black', label='Auxiliary Head')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/carlos/projects/TimeSeries/figure_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved architecture diagram")

# Figure 2: Data Preprocessing Pipeline
def create_preprocessing_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Step 1: Raw Data
    y_pos = 10.5
    step1_box = FancyBboxPatch((2, y_pos-0.6), 6, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#FFE6E6', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(5, y_pos, 'Step 1: Raw GLV Data', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(5, y_pos-0.35, 'Shape: (N, 7, 65) - 7 species over 65 time points',
            ha='center', va='center', fontsize=9)

    # Arrow
    arrow1 = FancyArrowPatch((5, y_pos-0.7), (5, y_pos-1.3), arrowstyle='->',
                            mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Step 2: Family Normalization
    y_pos = 8.5
    step2_box = FancyBboxPatch((1.5, y_pos-0.8), 7, 1.6, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#E6F3FF', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(5, y_pos+0.4, 'Step 2: Family-wise Normalization', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(5, y_pos, 'Find global max: max_family = max(X[n, :, :])',
            ha='center', va='center', fontsize=9, family='monospace')
    ax.text(5, y_pos-0.35, 'Normalize: X_family_norm = X / max_family',
            ha='center', va='center', fontsize=9, family='monospace')
    ax.text(5, y_pos-0.65, '→ Highest peak across all 7 species = 1.0',
            ha='center', va='center', fontsize=9, style='italic', color='darkblue')

    # Arrow
    arrow2 = FancyArrowPatch((5, y_pos-0.9), (5, y_pos-1.5), arrowstyle='->',
                            mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow2)

    # Step 3: Curve Sorting
    y_pos = 6.0
    step3_box = FancyBboxPatch((1.5, y_pos-0.8), 7, 1.6, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#E6FFE6', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(5, y_pos+0.4, 'Step 3: Curve Sorting by Peak', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(5, y_pos, 'max_per_curve = max(X_family_norm, axis=time)',
            ha='center', va='center', fontsize=9, family='monospace')
    ax.text(5, y_pos-0.35, 'sorted_indices = argsort(-max_per_curve)',
            ha='center', va='center', fontsize=9, family='monospace')
    ax.text(5, y_pos-0.65, '→ Consistent ordering: highest peak first',
            ha='center', va='center', fontsize=9, style='italic', color='darkgreen')

    # Arrow
    arrow3 = FancyArrowPatch((5, y_pos-0.9), (5, y_pos-1.5), arrowstyle='->',
                            mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow3)

    # Step 4: Individual Curve Normalization
    y_pos = 3.5
    step4_box = FancyBboxPatch((1.5, y_pos-0.8), 7, 1.6, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#FFF3E6', linewidth=2)
    ax.add_patch(step4_box)
    ax.text(5, y_pos+0.4, 'Step 4: Individual Curve Normalization', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(5, y_pos, 'max_per_curve = max(X_sorted, axis=time)',
            ha='center', va='center', fontsize=9, family='monospace')
    ax.text(5, y_pos-0.35, 'X_final = X_sorted / max_per_curve[:, None]',
            ha='center', va='center', fontsize=9, family='monospace')
    ax.text(5, y_pos-0.65, '→ Each curve peak normalized to 1.0',
            ha='center', va='center', fontsize=9, style='italic', color='darkorange')

    # Arrow
    arrow4 = FancyArrowPatch((5, y_pos-0.9), (5, y_pos-1.5), arrowstyle='->',
                            mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow4)

    # Final Output
    y_pos = 1.0
    output_box = FancyBboxPatch((1, y_pos-0.5), 8, 1.3, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#E6E6FA', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(5, y_pos+0.4, 'Final Data Package', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(2.5, y_pos-0.1, '• data: X_final (N, 7, 65)', ha='left', va='center', fontsize=9)
    ax.text(2.5, y_pos-0.35, '• reconstruction_max_values (N, 7)', ha='left', va='center', fontsize=9)
    ax.text(5.8, y_pos-0.1, '• family_max_values (N,)', ha='left', va='center', fontsize=9)

    # Title
    ax.text(5, 11.5, 'Data Preprocessing Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/carlos/projects/TimeSeries/figure_preprocessing.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved preprocessing pipeline diagram")

# Figure 3: Training Methodology
def create_training_methodology_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Title
    ax.text(5, 10.5, 'Training Methodology',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Loss Function Section
    y_pos = 9.2
    loss_box = FancyBboxPatch((0.5, y_pos-1.5), 9, 2.0, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#FFF0F0', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(5, y_pos+0.2, 'Loss Function', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(5, y_pos-0.2, '𝓛_total = 𝓛_recon + β·𝓛_KL + λ·𝓛_max_val',
            ha='center', va='center', fontsize=11, family='serif', style='italic')
    ax.text(1.8, y_pos-0.65, '𝓛_recon = MSE(X̂, X)', ha='left', va='center', fontsize=10, family='serif')
    ax.text(1.8, y_pos-1.0, '𝓛_KL = -0.5·Σ(1 + log σ² - μ² - σ²)', ha='left', va='center', fontsize=10, family='serif')
    ax.text(1.8, y_pos-1.35, '𝓛_max_val = MSE(max_pred[1:6], max_true[1:6])', ha='left', va='center', fontsize=10, family='serif')

    # Hyperparameters Section
    y_pos = 6.5
    hp_box = FancyBboxPatch((0.5, y_pos-2.3), 4.2, 2.8, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#F0F0FF', linewidth=2)
    ax.add_patch(hp_box)
    ax.text(2.6, y_pos+0.2, 'Hyperparameters', ha='center', va='center',
            fontsize=12, fontweight='bold')

    hp_text = [
        ('Learning Rate:', '1×10⁻⁴'),
        ('Batch Size:', '1000'),
        ('Epochs:', '2000'),
        ('β_max:', '2×10⁻⁴'),
        ('λ_max_val:', '0.5'),
        ('Warmup Epochs:', '300'),
        ('Latent Dim:', '30'),
        ('LSTM Hidden:', '256'),
        ('LSTM Layers:', '2')
    ]

    y_offset = y_pos - 0.2
    for i, (param, value) in enumerate(hp_text):
        ax.text(1.2, y_offset - i*0.32, param, ha='left', va='center', fontsize=9)
        ax.text(3.8, y_offset - i*0.32, value, ha='right', va='center', fontsize=9,
                fontweight='bold', family='monospace')

    # Training Schedule Section
    y_pos = 6.5
    schedule_box = FancyBboxPatch((5.3, y_pos-2.3), 4.2, 2.8, boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='#F0FFF0', linewidth=2)
    ax.add_patch(schedule_box)
    ax.text(7.4, y_pos+0.2, 'Training Schedule', ha='center', va='center',
            fontsize=12, fontweight='bold')

    schedule_text = [
        'β Warmup:',
        '  • Linear: 0 → β_max',
        '  • Duration: 300 epochs',
        '',
        'Teacher Forcing Decay:',
        '  • Exponential: 1.0 → 0.025',
        '  • Duration: 40% of epochs',
        '',
        'Optimizer: Adam'
    ]

    y_offset = y_pos - 0.2
    for i, text in enumerate(schedule_text):
        ax.text(5.8, y_offset - i*0.28, text, ha='left', va='center', fontsize=9)

    # Data Configuration Section
    y_pos = 2.8
    data_box = FancyBboxPatch((0.5, y_pos-1.8), 9, 2.3, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#FFFEF0', linewidth=2)
    ax.add_patch(data_box)
    ax.text(5, y_pos+0.5, 'Data Configuration', ha='center', va='center',
            fontsize=12, fontweight='bold')

    data_text = [
        'Input Shape: (N, 7, 65) - 7 species time series with 65 time points',
        'Target: Reconstruction of normalized curves + prediction of max values',
        'Max Value Prediction: 6 values (curves 1-6), curve 0 fixed at 1.0',
        'Scale Prediction Mode: Logarithmic transform for numerical stability',
        'Evaluation Metrics: Reconstruction MSE, KL divergence, Max value MSE, Coverage (±2σ)'
    ]

    y_offset = y_pos + 0.1
    for i, text in enumerate(data_text):
        ax.text(5, y_offset - i*0.35, text, ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/carlos/projects/TimeSeries/figure_training_methodology.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved training methodology diagram")

# Figure 4: Loss Components Diagram
def create_loss_diagram():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Simulate training curves for illustration
    epochs = np.arange(0, 2000, 20)

    # Reconstruction loss - decreasing
    recon_loss = 0.15 * np.exp(-epochs/400) + 0.005 + 0.002*np.random.randn(len(epochs))*np.exp(-epochs/300)

    # KL divergence - increases then stabilizes
    kl_loss = 2e-4 * (1 - np.exp(-epochs/300)) * (1 + 0.1*np.sin(epochs/100)) + 1e-5*np.random.randn(len(epochs))
    kl_loss = np.maximum(kl_loss, 0)

    # Max value loss - decreasing
    max_val_loss = 0.08 * np.exp(-epochs/350) + 0.002 + 0.001*np.random.randn(len(epochs))*np.exp(-epochs/250)

    # Total loss
    total_loss = recon_loss + 2e-4 * kl_loss * np.minimum(epochs/300, 1) + 0.5 * max_val_loss

    # Plot reconstruction loss
    axes[0, 0].plot(epochs, recon_loss, linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Reconstruction Loss', fontsize=11)
    axes[0, 0].set_title('Reconstruction Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(bottom=0)

    # Plot KL divergence
    axes[0, 1].plot(epochs, kl_loss, linewidth=2, color='#A23B72')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('KL Divergence', fontsize=11)
    axes[0, 1].set_title('KL Divergence (with β warmup)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)

    # Plot max value loss
    axes[1, 0].plot(epochs, max_val_loss, linewidth=2, color='#F18F01')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Max Value Loss', fontsize=11)
    axes[1, 0].set_title('Max Value Prediction Loss', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)

    # Plot total loss
    axes[1, 1].plot(epochs, total_loss, linewidth=2, color='#06A77D')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Total Loss', fontsize=11)
    axes[1, 1].set_title('Total Combined Loss', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)

    plt.suptitle('Training Loss Components (Illustrative)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/home/carlos/projects/TimeSeries/figure_loss_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved loss components diagram")

if __name__ == "__main__":
    print("Generating methodology figures...")
    create_architecture_diagram()
    create_preprocessing_diagram()
    create_training_methodology_diagram()
    create_loss_diagram()
    print("\n✓ All figures generated successfully!")
