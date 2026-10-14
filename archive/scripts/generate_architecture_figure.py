"""
Professional LSTM-VAE architecture figure for academic publication.
Minimalist, clean design following top-venue standards.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe

# Publication settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Minimal professional palette
C = {
    'enc': '#2E86AB',    # Blue
    'lat': '#F18F01',    # Orange
    'dec': '#C73E1D',    # Red
    'aux': '#6A994E',    # Green
    'gray': '#495057',   # Dark gray
    'light': '#ADB5BD',  # Light gray
}

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6.4)
ax.axis('off')
ax.set_aspect('equal')

def block(x, y, w, h, label, sublabel, color):
    """Clean rectangular block with label."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor='white', linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.15, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    if sublabel:
        ax.text(x, y - 0.25, sublabel, ha='center', va='center',
                fontsize=7, color='white', alpha=0.9)

def small_block(x, y, w, h, label, color):
    """Small block."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='white', linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, fontweight='medium', color='white')

def node(x, y, r, label, color):
    """Circular node."""
    c = Circle((x, y), r, facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(c)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

def arrow(start, end, color='#343A40'):
    """Clean arrow."""
    arr = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        color=color,
        linewidth=1.8,
        mutation_scale=14,
        shrinkA=2, shrinkB=2
    )
    ax.add_patch(arr)

def curved_arrow(start, end, rad, color='#343A40'):
    """Curved arrow."""
    arr = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        connectionstyle=f'arc3,rad={rad}',
        color=color,
        linewidth=1.8,
        mutation_scale=14,
        shrinkA=2, shrinkB=2
    )
    ax.add_patch(arr)

# Positions
y_mid = 3.2
y_top = 4.6
y_bot = 1.8

# === ENCODER SECTION ===
# Input
ax.text(1.5, y_mid, r'$\mathbf{X}$', ha='center', va='center',
        fontsize=12, fontweight='bold', color=C['gray'])
ax.text(1.5, y_mid - 0.6, '(N, 7, 65)', ha='center', va='center',
        fontsize=7, color=C['light'])

# Encoder block
block(4, y_mid, 2.2, 1.4, 'Encoder', 'Bi-LSTM', C['enc'])

arrow((2.2, y_mid), (2.9, y_mid))

# === LATENT SECTION ===
# μ and σ branches
small_block(6.5, y_top, 1.0, 0.7, r'$\boldsymbol{\mu}$', C['lat'])
small_block(6.5, y_bot, 1.0, 0.7, r'$\boldsymbol{\sigma}$', C['lat'])

curved_arrow((5.1, y_mid + 0.4), (6.0, y_top), -0.15)
curved_arrow((5.1, y_mid - 0.4), (6.0, y_bot), 0.15)

# Latent z
node(8.2, y_mid, 0.45, r'$\mathbf{z}$', C['lat'])

curved_arrow((7.0, y_top), (7.8, y_mid + 0.35), -0.15)
curved_arrow((7.0, y_bot), (7.8, y_mid - 0.35), 0.15)

# Sampling notation
ax.text(7.35, y_mid, r'$\sim$', ha='center', va='center',
        fontsize=10, color=C['gray'])

# === DECODER SECTION ===
# Decoder block
block(11, y_mid, 2.2, 1.4, 'Decoder', 'LSTM', C['dec'])

arrow((8.65, y_mid), (9.9, y_mid))

# Output
ax.text(14, y_mid, r'$\hat{\mathbf{X}}$', ha='center', va='center',
        fontsize=12, fontweight='bold', color=C['gray'])
ax.text(14, y_mid - 0.6, '(N, 7, 65)', ha='center', va='center',
        fontsize=7, color=C['light'])

arrow((12.1, y_mid), (13.3, y_mid))

# === SCALE PREDICTION BRANCH ===
# MLP block
block(11, y_bot - 0.6, 1.8, 0.9, 'MLP', None, C['aux'])

# Scale output
ax.text(14, y_bot - 0.6, r'$\hat{\mathbf{m}}$', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C['gray'])
ax.text(14, y_bot - 1.15, '(N, 7)', ha='center', va='center',
        fontsize=7, color=C['light'])

# z to MLP
curved_arrow((8.2, y_mid - 0.45), (10.1, y_bot - 0.6), 0.25)

# MLP to output
arrow((11.9, y_bot - 0.6), (13.3, y_bot - 0.6))

# === SECTION LABELS ===
ax.text(4, 5.8, 'Encoder', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C['enc'])
ax.text(7.35, 5.8, 'Latent Space', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C['lat'])
ax.text(11, 5.8, 'Decoder', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C['dec'])
ax.text(11, 0.4, 'Scale Predictor', ha='center', va='center',
        fontsize=9, fontweight='bold', color=C['aux'])

# === LOSS ANNOTATIONS (subtle) ===
ax.text(6.5, y_mid, r'$\mathcal{L}_{KL}$', ha='center', va='center',
        fontsize=7, color=C['light'], alpha=0.8)

plt.tight_layout(pad=0.3)
plt.savefig('figure_architecture_cvae.pdf', format='pdf', bbox_inches='tight', pad_inches=0.08)
plt.savefig('figure_architecture_cvae.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
print("Saved: figure_architecture_cvae.pdf and figure_architecture_cvae.png")
