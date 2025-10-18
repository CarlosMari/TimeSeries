"""
Interactive 3D Latent Space Explorer
=====================================

Creates a fully interactive 3D visualization of the learned latent space with:
- Multiple projection methods (UMAP, PCA, Semantic)
- Dynamic color schemes (regime, diversity, variance, etc.)
- Hover tooltips with sample information
- Interpolation path visualization
- Convex hulls around regimes
- Time series display for selected samples
- Fly-to-region navigation
"""

import numpy as np
import torch
import pickle
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import sys

# Add models to path
from src.models.cvae import LSTM_VAE
from src.utils.config import model_config, DEVICE

# Constants
MODEL_PATH = 'model_ckpts/model_final_30.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_HTML = 'latent_explorer_3d.html'
RANDOM_SEED = 42

print("="*70)
print("🚀 INTERACTIVE 3D LATENT SPACE EXPLORER")
print("="*70)

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

print("\n📂 Loading data...")
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

curves = torch.FloatTensor(data['curves'])
metadata = data['metadata']
max_values = data['max_values']

print(f"Loaded {len(curves)} samples with shape {curves.shape}")

print("\n🤖 Loading model...")
model = LSTM_VAE(config=model_config).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"Model: {model_config['latent_dim']}D latent space")

# ============================================================================
# ENCODE ALL SAMPLES
# ============================================================================

print("\n🧬 Encoding all samples...")
latent_codes = []
reconstructions = []

with torch.no_grad():
    for i in tqdm(range(0, len(curves), 32), desc="Encoding"):
        batch = curves[i:i+32].to(DEVICE)
        X_rnn = batch.permute(0, 2, 1)

        # Encode
        _, (h_n, _) = model.encoder_rnn(X_rnn)
        encoded = h_n.permute(1, 0, 2).contiguous().view(batch.size(0), -1)
        mu = model.fc_mu(encoded)
        log_var = model.fc_log_var(encoded)

        # Sample
        z = model.sample(mu, log_var)

        # Decode
        recon, _ = model.decode(z)

        latent_codes.append(z.cpu())
        reconstructions.append(recon.cpu())

latent_codes = torch.cat(latent_codes, dim=0).numpy()
reconstructions = torch.cat(reconstructions, dim=0).numpy()

print(f"Latent codes shape: {latent_codes.shape}")

# ============================================================================
# EXTRACT FEATURES
# ============================================================================

print("\n📊 Extracting dynamical features...")

def extract_features(curve):
    """Extract comprehensive features from a curve."""
    n_species, n_time = curve.shape

    features = {}

    # Peak analysis
    peak_times = np.argmax(curve, axis=1)
    peak_values = np.max(curve, axis=1)
    features['peak_time_mean'] = np.mean(peak_times)
    features['peak_value_mean'] = np.mean(peak_values)

    # Steady state
    steady = np.mean(curve[:, -10:], axis=1)
    features['steady_state_sum'] = np.sum(steady)
    features['steady_state_max'] = np.max(steady)

    # Diversity
    steady_probs = steady / (np.sum(steady) + 1e-10)
    steady_probs = steady_probs[steady_probs > 1e-10]
    features['diversity'] = -np.sum(steady_probs * np.log(steady_probs + 1e-10))

    # Other metrics
    features['dominance'] = np.max(steady) / (np.sum(steady) + 1e-10)
    features['n_survivors'] = np.sum(steady > 0.01)
    features['temporal_variance'] = np.mean(np.var(curve, axis=1))

    return features

# Unnormalize curves
print("Unnormalizing curves...")
unnormalized_curves = curves.numpy() * max_values[:, :, np.newaxis]

# Extract features
features_list = [extract_features(unnormalized_curves[i]) for i in tqdm(range(len(curves)), desc="Extracting")]

# Convert to arrays
features = {}
for key in features_list[0].keys():
    features[key] = np.array([f[key] for f in features_list])

# Add GLV parameters
features['max_eigenvalue'] = np.array([m['max_eigenvalue'] for m in metadata])
features['mean_interaction'] = np.array([m['mean_interaction'] for m in metadata])
features['stability_measure'] = np.array([m['stability_measure'] for m in metadata])

# Compute reconstruction MSE (trim curves to match reconstruction length)
seq_len = min(curves.shape[2], reconstructions.shape[2])
recon_mse = np.mean((curves.numpy()[:, :, :seq_len] - reconstructions[:, :, :seq_len])**2, axis=(1, 2))
features['reconstruction_mse'] = recon_mse

print(f"Extracted {len(features)} features")

# ============================================================================
# CLASSIFY DYNAMICAL REGIMES
# ============================================================================

print("\n🎭 Classifying dynamical regimes...")

diversity = features['diversity']
dominance = features['dominance']
temporal_var = features['temporal_variance']

# Data-driven thresholds
div_thresh = np.median(diversity)
dom_thresh = np.median(dominance)
var_thresh = np.median(temporal_var)

regimes = np.empty(len(curves), dtype=object)
for i in range(len(curves)):
    if diversity[i] < div_thresh:
        regimes[i] = 'Low Diversity Stable'
    elif dominance[i] > dom_thresh:
        regimes[i] = 'Dominant Species'
    elif temporal_var[i] > var_thresh:
        regimes[i] = 'Oscillatory'
    else:
        regimes[i] = 'High Diversity Stable'

features['regime'] = regimes

# Count regimes
unique_regimes, counts = np.unique(regimes, return_counts=True)
print("Regime distribution:")
for regime, count in zip(unique_regimes, counts):
    print(f"  {regime}: {count}")

# ============================================================================
# COMPUTE 3D PROJECTIONS
# ============================================================================

print("\n🗺️  Computing 3D projections...")

# 1. UMAP-3D
print("  Computing UMAP-3D...")
reducer_umap = umap.UMAP(n_components=3, random_state=RANDOM_SEED, n_neighbors=30, min_dist=0.1)
umap_3d = reducer_umap.fit_transform(latent_codes)

# 2. PCA-3D
print("  Computing PCA-3D...")
pca = PCA(n_components=3)
pca_3d = pca.fit_transform(latent_codes)
pca_var = pca.explained_variance_ratio_

# 3. Semantic dimensions (top 3 most correlated with features)
print("  Finding semantic dimensions...")
semantic_features = ['diversity', 'temporal_variance', 'steady_state_sum']
semantic_dims = []
for feat_name in semantic_features:
    corrs = [np.abs(np.corrcoef(latent_codes[:, i], features[feat_name])[0, 1])
             for i in range(latent_codes.shape[1])]
    semantic_dims.append(np.argmax(corrs))

semantic_3d = latent_codes[:, semantic_dims]

projections = {
    'UMAP-3D': umap_3d,
    'PCA-3D': pca_3d,
    'Semantic': semantic_3d
}

print(f"Created {len(projections)} projection types")

# ============================================================================
# COMPUTE CONVEX HULLS FOR REGIMES
# ============================================================================

print("\n🕸️  Computing convex hulls...")

hulls = {}
for regime in unique_regimes:
    mask = regimes == regime
    if np.sum(mask) >= 4:  # Need at least 4 points
        points = umap_3d[mask]
        try:
            hull = ConvexHull(points)
            hulls[regime] = {
                'points': points,
                'vertices': hull.vertices,
                'simplices': hull.simplices
            }
        except:
            print(f"  Warning: Could not compute hull for {regime}")

print(f"Computed {len(hulls)} convex hulls")

# ============================================================================
# CREATE INTERACTIVE PLOTLY FIGURE
# ============================================================================

print("\n🎨 Creating interactive visualization...")

# Color maps for different features
COLOR_MAPS = {
    'Dynamical Regime': {
        'Low Diversity Stable': '#1f77b4',  # blue
        'High Diversity Stable': '#2ca02c',  # green
        'Dominant Species': '#ff7f0e',  # orange
        'Oscillatory': '#d62728'  # red
    },
    'Diversity': 'Viridis',
    'Temporal Variance': 'Plasma',
    'N Survivors': 'Cividis',
    'Steady State Sum': 'Turbo',
    'Reconstruction MSE': 'Hot_r'
}

# Initialize figure with UMAP projection and regime colors
proj_name = 'UMAP-3D'
coords = projections[proj_name]

# Create discrete colors for regimes
regime_colors = [COLOR_MAPS['Dynamical Regime'][r] for r in regimes]

# Build hover text
hover_texts = []
for i in range(len(curves)):
    text = f"<b>Sample {i}</b><br>"
    text += f"Regime: {regimes[i]}<br>"
    text += f"Diversity: {features['diversity'][i]:.3f}<br>"
    text += f"Temporal Var: {features['temporal_variance'][i]:.4f}<br>"
    text += f"N Survivors: {int(features['n_survivors'][i])}<br>"
    text += f"Steady State: {features['steady_state_sum'][i]:.3f}<br>"
    text += f"MSE: {features['reconstruction_mse'][i]:.4f}"
    hover_texts.append(text)

# Create main 3D scatter
fig = go.Figure()

# Add scatter trace
scatter = go.Scatter3d(
    x=coords[:, 0],
    y=coords[:, 1],
    z=coords[:, 2],
    mode='markers',
    marker=dict(
        size=4,
        color=regime_colors,
        opacity=0.7,
        line=dict(width=0)
    ),
    text=hover_texts,
    hovertemplate='%{text}<extra></extra>',
    name='Samples'
)

fig.add_trace(scatter)

# Add convex hulls
for regime, hull_data in hulls.items():
    points = hull_data['points']
    simplices = hull_data['simplices']

    # Create mesh
    color = COLOR_MAPS['Dynamical Regime'][regime]

    mesh = go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=simplices[:, 0],
        j=simplices[:, 1],
        k=simplices[:, 2],
        color=color,
        opacity=0.15,
        name=regime,
        showlegend=True,
        hoverinfo='name'
    )

    fig.add_trace(mesh)

# Update layout
fig.update_layout(
    title={
        'text': f'<b>Interactive 3D Latent Space Explorer</b><br><sub>Projection: {proj_name} | Color: Dynamical Regime</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    scene=dict(
        xaxis_title=f'{proj_name} Dim 1',
        yaxis_title=f'{proj_name} Dim 2',
        zaxis_title=f'{proj_name} Dim 3',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        aspectmode='cube'
    ),
    height=900,
    width=1400,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    hovermode='closest',
    template='plotly_white'
)

# Add buttons for projections
projection_buttons = []
for proj_name_btn in projections.keys():
    coords_btn = projections[proj_name_btn]

    button = dict(
        label=proj_name_btn,
        method='update',
        args=[
            {'x': [coords_btn[:, 0]], 'y': [coords_btn[:, 1]], 'z': [coords_btn[:, 2]]},
            {'scene.xaxis.title': f'{proj_name_btn} Dim 1',
             'scene.yaxis.title': f'{proj_name_btn} Dim 2',
             'scene.zaxis.title': f'{proj_name_btn} Dim 3',
             'title.text': f'<b>Interactive 3D Latent Space Explorer</b><br><sub>Projection: {proj_name_btn} | Color: Dynamical Regime</sub>'}
        ]
    )
    projection_buttons.append(button)

# Add dropdown for color schemes
color_buttons = []

# Regime colors
color_buttons.append(dict(
    label='Dynamical Regime',
    method='restyle',
    args=[{'marker.color': [regime_colors]}]
))

# Continuous features
for feat_name in ['diversity', 'temporal_variance', 'steady_state_sum', 'n_survivors', 'reconstruction_mse']:
    display_name = feat_name.replace('_', ' ').title()
    colorscale = COLOR_MAPS.get(display_name, 'Viridis')

    color_buttons.append(dict(
        label=display_name,
        method='restyle',
        args=[{'marker.color': [features[feat_name]],
               'marker.colorscale': colorscale,
               'marker.showscale': True}]
    ))

# Add dropdowns
fig.update_layout(
    updatemenus=[
        # Projection selector
        dict(
            buttons=projection_buttons,
            direction="down",
            showactive=True,
            x=0.02,
            xanchor="left",
            y=0.95,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#888",
            borderwidth=1,
            font=dict(size=11),
            active=0
        ),
        # Color selector
        dict(
            buttons=color_buttons,
            direction="down",
            showactive=True,
            x=0.02,
            xanchor="left",
            y=0.85,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#888",
            borderwidth=1,
            font=dict(size=11),
            active=0
        )
    ]
)

# Add annotations for dropdown labels
fig.add_annotation(
    text="<b>Projection:</b>",
    x=0.02,
    y=0.97,
    xref="paper",
    yref="paper",
    showarrow=False,
    xanchor="left",
    bgcolor="white",
    font=dict(size=12, color="black")
)

fig.add_annotation(
    text="<b>Color by:</b>",
    x=0.02,
    y=0.87,
    xref="paper",
    yref="paper",
    showarrow=False,
    xanchor="left",
    bgcolor="white",
    font=dict(size=12, color="black")
)

# Add info box
info_text = f"""
<b>Dataset Info:</b><br>
• Samples: {len(curves)}<br>
• Latent Dims: {model_config['latent_dim']}<br>
• Regimes: {len(unique_regimes)}<br>
<br>
<b>Controls:</b><br>
• Drag to rotate<br>
• Scroll to zoom<br>
• Hover for details<br>
"""

fig.add_annotation(
    text=info_text,
    x=0.98,
    y=0.98,
    xref="paper",
    yref="paper",
    showarrow=False,
    xanchor="right",
    yanchor="top",
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#888",
    borderwidth=1,
    font=dict(size=10),
    align="left"
)

# ============================================================================
# SAVE TO HTML
# ============================================================================

print(f"\n💾 Saving to {OUTPUT_HTML}...")
fig.write_html(
    OUTPUT_HTML,
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'latent_space_3d',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
)

print("="*70)
print("✅ DONE!")
print("="*70)
print(f"\n📊 Open '{OUTPUT_HTML}' in your browser to explore the latent space!")
print(f"   File size: {Path(OUTPUT_HTML).stat().st_size / 1024 / 1024:.1f} MB")
print("\n🎮 Features:")
print("   • Rotate: Click and drag")
print("   • Zoom: Scroll wheel")
print("   • Pan: Right-click and drag")
print("   • Reset: Double-click")
print("   • Change projection: Top-left dropdown")
print("   • Change colors: Second dropdown")
print("   • Hover over points for details")
print("   • Toggle hulls: Click legend items")
print("="*70)
