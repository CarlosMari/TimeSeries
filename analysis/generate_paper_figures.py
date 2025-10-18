"""
SOTA Latent Space Analysis for VAE on GLV Time Series

Implements state-of-the-art analysis techniques from:
- β-VAE: Learning Basic Visual Concepts (Higgins et al., 2017)
- Understanding disentangling in β-VAE (Burgess et al., 2018)
- Disentangling by Factorising (Kim & Mnih, 2018)
- Challenging Common Assumptions (Locatello et al., 2019)

Generates comprehensive publication-quality figures covering:
1. Latent space structure and organization
2. Disentanglement metrics and visualization
3. Generative capabilities and interpolations
4. Manifold topology and geometry
5. Capacity utilization and information content
6. Dynamics encoding and semantic control
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap

from src.models.cvae import LSTM_VAE
from src.utils.config import model_config, DEVICE

# High-quality plotting settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'figure.dpi': 100,
})

# Configuration
MODEL_PATH = 'model_ckpts/model_final_30.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_DIR = Path("sota_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Analysis parameters
N_SAMPLES = 10000  # Number of samples for UMAP/t-SNE
N_TRAVERSALS = 8   # Number of latent dimensions to traverse
N_INTERPOLATIONS = 5  # Number of interpolation examples
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_dynamical_features(curve):
    """Extract comprehensive dynamical features from a time series curve."""
    n_species, n_time = curve.shape

    features = {}

    # Peak analysis
    features['peak_times'] = np.argmax(curve, axis=1)
    features['peak_values'] = np.max(curve, axis=1)
    features['peak_time_mean'] = np.mean(features['peak_times'])
    features['peak_time_std'] = np.std(features['peak_times'])

    # Steady state analysis (last 10% of time)
    steady_window = max(10, n_time // 10)
    features['steady_states'] = np.mean(curve[:, -steady_window:], axis=1)
    features['steady_state_mean'] = np.mean(features['steady_states'])
    features['steady_state_sum'] = np.sum(features['steady_states'])

    # Overshoot
    steady_safe = np.maximum(features['steady_states'], 1e-3)
    features['overshoots'] = (features['peak_values'] - steady_safe) / steady_safe
    features['overshoot_mean'] = np.mean(features['overshoots'])

    # Temporal dynamics
    features['temporal_variance'] = np.mean(np.var(curve, axis=1))
    features['temporal_range'] = np.mean(np.ptp(curve, axis=1))

    # Initial growth (slope in first 20% of time)
    growth_window = max(5, n_time // 5)
    features['initial_growth'] = np.mean(np.gradient(curve[:, :growth_window], axis=1))

    # Survivors (species with steady state > threshold)
    features['n_survivors'] = np.sum(features['steady_states'] > 0.01)

    # Diversity (Shannon entropy of steady states)
    steady_probs = features['steady_states'] / (features['steady_state_sum'] + 1e-10)
    steady_probs = steady_probs[steady_probs > 1e-10]
    features['diversity'] = -np.sum(steady_probs * np.log(steady_probs + 1e-10))

    # Dominance (max steady state / sum)
    features['dominance'] = np.max(features['steady_states']) / (features['steady_state_sum'] + 1e-10)

    return features


def compute_kl_per_dimension(mu, log_var):
    """Compute KL divergence for each latent dimension."""
    # KL(q(z|x) || p(z)) for each dimension
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl_per_dim.mean(dim=0).cpu().numpy()  # Average over batch


def compute_mutual_information(latent_codes, features, n_bins=20):
    """Compute mutual information between latent dimensions and features."""
    n_latent = latent_codes.shape[1]
    n_features = len(features)

    mi_matrix = np.zeros((n_latent, n_features))

    feature_names = list(features.keys())

    for i in tqdm(range(n_latent), desc="Computing MI"):
        latent_binned = np.digitize(latent_codes[:, i],
                                     bins=np.linspace(latent_codes[:, i].min(),
                                                     latent_codes[:, i].max(),
                                                     n_bins))

        for j, feat_name in enumerate(feature_names):
            feat_vals = features[feat_name]
            feat_binned = np.digitize(feat_vals,
                                     bins=np.linspace(feat_vals.min(),
                                                     feat_vals.max(),
                                                     n_bins))

            mi_matrix[i, j] = mutual_info_score(latent_binned, feat_binned)

    return mi_matrix, feature_names


def compute_sap_score(latent_codes, features):
    """
    Compute SAP (Separated Attribute Predictability) score.
    Measures how well each latent dimension predicts a single feature.
    """
    n_latent = latent_codes.shape[1]
    feature_names = list(features.keys())
    n_features = len(feature_names)

    # Matrix of R² scores
    r2_matrix = np.zeros((n_latent, n_features))

    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_codes)

    for j, feat_name in enumerate(tqdm(feature_names, desc="Computing SAP")):
        y = features[feat_name].reshape(-1, 1)

        for i in range(n_latent):
            X = latent_scaled[:, i].reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)
            r2_matrix[i, j] = model.score(X, y)

    # SAP score: For each feature, difference between top 2 R² scores
    sap_scores = []
    for j in range(n_features):
        sorted_r2 = np.sort(r2_matrix[:, j])[::-1]
        if len(sorted_r2) >= 2:
            sap_scores.append(sorted_r2[0] - sorted_r2[1])
        else:
            sap_scores.append(0)

    return r2_matrix, sap_scores, feature_names


def latent_arithmetic(model, z1, z2=None, z3=None, alpha=1.0, device=DEVICE):
    """
    Perform latent arithmetic: z_new = z1 + alpha * (z2 - z3)
    If z2 is None, just return z1 (identity)
    If z3 is None, do: z_new = z1 + alpha * z2

    Returns:
        decoded (np.ndarray): Generated sequences (normalized)
        max_vals (np.ndarray): Predicted max values for unnormalization
        z_new (torch.Tensor): The resulting latent code
    """
    if z2 is None:
        z_new = z1
    elif z3 is None:
        z_new = z1 + alpha * z2
    else:
        z_new = z1 + alpha * (z2 - z3)

    with torch.no_grad():
        decoded, max_vals = model.decode(z_new.to(device))

    return decoded.cpu().numpy(), max_vals.cpu().numpy(), z_new


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def load_data_and_encode():
    """Load data, model, and encode all samples."""
    print("="*70)
    print("LOADING DATA AND MODEL")
    print("="*70)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    curves = torch.tensor(data['curves'], dtype=torch.float32)
    metadata = data['metadata']

    print(f"Loaded {len(curves)} samples")
    print(f"Curve shape: {curves[0].shape}")

    # Extract properties from metadata
    properties = {}
    property_keys = ['max_eigenvalue', 'mean_interaction', 'interaction_std',
                     'diagonal_strength', 'stability_measure']

    for key in property_keys:
        properties[key] = np.array([m[key] for m in metadata])

    # Extract steady state info
    properties['steady_state_sum'] = np.array([np.sum(m['steady_state']) for m in metadata])

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = LSTM_VAE(config=model_config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully")

    # Encode all samples
    print("\nEncoding all samples...")
    latent_codes = []
    latent_mu = []
    latent_log_var = []
    reconstructions = []

    with torch.no_grad():
        batch_size = 64
        for i in tqdm(range(0, len(curves), batch_size)):
            batch = curves[i:i+batch_size].to(DEVICE)
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
            latent_mu.append(mu.cpu())
            latent_log_var.append(log_var.cpu())
            reconstructions.append(recon.cpu())

    latent_codes = torch.cat(latent_codes, dim=0).numpy()
    latent_mu = torch.cat(latent_mu, dim=0)
    latent_log_var = torch.cat(latent_log_var, dim=0)
    reconstructions = torch.cat(reconstructions, dim=0).numpy()

    print(f"\nEncoding complete!")
    print(f"Latent codes shape: {latent_codes.shape}")

    # Unnormalize curves using max_values
    print("\nUnnormalizing curves...")
    max_vals = data['max_values']  # Shape: (N, 7)
    unnormalized_curves = curves * torch.tensor(max_vals, dtype=torch.float32).unsqueeze(-1)
    print(f"Unnormalized curves shape: {unnormalized_curves.shape}")
    print(f"Sample unnormalized max values: {unnormalized_curves[0].max(dim=1)[0][:3]}")

    # Extract features from UNNORMALIZED curves
    print("\nExtracting dynamical features from unnormalized curves...")
    features_list = [extract_dynamical_features(unnormalized_curves[i].numpy()) for i in tqdm(range(len(unnormalized_curves)))]

    # Organize features into arrays
    features = {}
    for key in features_list[0].keys():
        features[key] = np.array([f[key] for f in features_list])

    # Add interaction properties (already in array format)
    print("\nAdding interaction properties...")
    for key, value in properties.items():
        features[key] = value

    return {
        'model': model,
        'curves': curves.numpy(),
        'reconstructions': reconstructions,
        'latent_codes': latent_codes,
        'latent_mu': latent_mu,
        'latent_log_var': latent_log_var,
        'features': features,
        'properties': properties,
    }


# ============================================================================
# FIGURE 1: LATENT SPACE STRUCTURE
# ============================================================================

def generate_figure1_latent_structure(data):
    """
    Comprehensive latent space structure visualization.

    Panel A: UMAP colored by peak time
    Panel B: UMAP colored by steady state sum
    Panel C: UMAP colored by diversity
    Panel D: UMAP colored by survivors
    Panel E: KL divergence per dimension
    Panel F: Posterior mean/std per dimension
    """
    print("\n" + "="*70)
    print("FIGURE 1: LATENT SPACE STRUCTURE")
    print("="*70)

    latent_codes = data['latent_codes']
    features = data['features']

    # Subsample for UMAP
    if len(latent_codes) > N_SAMPLES:
        idx = np.random.choice(len(latent_codes), N_SAMPLES, replace=False)
        latent_subset = latent_codes[idx]
        features_subset = {k: v[idx] for k, v in features.items()}
    else:
        latent_subset = latent_codes
        features_subset = features

    # Compute UMAP (3 components, will plot 2 vs 3)
    print("\nComputing UMAP (3 components)...")
    reducer = umap.UMAP(n_components=3, random_state=RANDOM_SEED, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(latent_subset)

    # Compute KL per dimension
    print("Computing KL per dimension...")
    kl_per_dim = compute_kl_per_dimension(data['latent_mu'], data['latent_log_var'])

    # Compute posterior statistics
    mu_mean = data['latent_mu'].mean(dim=0).numpy()
    mu_std = data['latent_mu'].std(dim=0).numpy()
    logvar_mean = data['latent_log_var'].mean(dim=0).numpy()

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # UMAP panels
    umap_features = [
        ('peak_time_mean', 'Mean Peak Time', 'viridis'),
        ('steady_state_sum', 'Total Steady State', 'plasma'),
        ('diversity', 'Shannon Diversity', 'cividis'),
        ('n_survivors', 'Number of Survivors', 'Spectral'),
    ]

    for i, (feat_name, title, cmap) in enumerate(umap_features):
        ax = fig.add_subplot(gs[0, i])

        # Plot UMAP 2 vs UMAP 3 (indices 1 and 2)
        scatter = ax.scatter(embedding[:, 1], embedding[:, 2],
                           c=features_subset[feat_name], cmap=cmap,
                           s=15, alpha=0.6, edgecolors='none')

        ax.set_xlabel('UMAP 2', fontweight='bold')
        ax.set_ylabel('UMAP 3', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.grid(alpha=0.2)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(title, fontsize=9)

    # Panel E: KL per dimension
    ax_kl = fig.add_subplot(gs[1, 0:2])

    dims = np.arange(len(kl_per_dim))
    colors = plt.cm.viridis(kl_per_dim / kl_per_dim.max())

    bars = ax_kl.bar(dims, kl_per_dim, color=colors, edgecolor='black', linewidth=0.5)
    ax_kl.set_xlabel('Latent Dimension', fontweight='bold')
    ax_kl.set_ylabel('KL Divergence', fontweight='bold')
    ax_kl.set_title('Information Content per Dimension', fontweight='bold', fontsize=13)
    ax_kl.grid(alpha=0.2, axis='y')

    # Add statistics
    active_dims = np.sum(kl_per_dim > 0.01)
    textstr = f'Active dims (KL>0.01): {active_dims}/{len(kl_per_dim)}\n'
    textstr += f'Mean KL: {kl_per_dim.mean():.3f}\n'
    textstr += f'Total KL: {kl_per_dim.sum():.2f}'
    ax_kl.text(0.98, 0.97, textstr, transform=ax_kl.transAxes,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
              fontsize=10)

    # Panel F: Posterior statistics
    ax_post = fig.add_subplot(gs[1, 2:4])

    x = np.arange(len(mu_mean))
    ax_post.errorbar(x, mu_mean, yerr=mu_std, fmt='o', markersize=4,
                    capsize=3, capthick=1, color='steelblue', alpha=0.7,
                    label='μ ± σ')
    ax_post.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax_post.set_xlabel('Latent Dimension', fontweight='bold')
    ax_post.set_ylabel('Posterior Mean', fontweight='bold')
    ax_post.set_title('Posterior Statistics Across Dataset', fontweight='bold', fontsize=13)
    ax_post.legend(loc='upper right')
    ax_post.grid(alpha=0.2)

    plt.savefig(OUTPUT_DIR / 'Figure1_Latent_Structure.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure1_Latent_Structure.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 1")
    plt.close()


# ============================================================================
# FIGURE 2: DISENTANGLEMENT ANALYSIS
# ============================================================================

def generate_figure2_disentanglement(data):
    """
    Comprehensive disentanglement analysis.

    Panel A: Latent traversals (6-8 key dimensions)
    Panel B: Correlation heatmap (latent dims vs features)
    Panel C: SAP scores per feature
    Panel D: Top latent-feature associations
    """
    print("\n" + "="*70)
    print("FIGURE 2: DISENTANGLEMENT ANALYSIS")
    print("="*70)

    model = data['model']
    latent_codes = data['latent_codes']
    features = data['features']

    # Select interesting dimensions (highest KL)
    kl_per_dim = compute_kl_per_dimension(data['latent_mu'], data['latent_log_var'])
    top_dims = np.argsort(kl_per_dim)[::-1][:N_TRAVERSALS]

    # Compute correlations
    print("\nComputing correlations...")
    feature_names = ['peak_time_mean', 'steady_state_sum', 'overshoot_mean',
                    'diversity', 'n_survivors', 'dominance', 'temporal_variance']

    correlations = np.zeros((len(latent_codes[0]), len(feature_names)))
    for i in range(len(latent_codes[0])):
        for j, fname in enumerate(feature_names):
            correlations[i, j] = np.corrcoef(latent_codes[:, i], features[fname])[0, 1]

    # Compute SAP scores
    print("\nComputing SAP scores...")
    features_sap = {k: features[k] for k in feature_names}
    r2_matrix, sap_scores, _ = compute_sap_score(latent_codes, features_sap)

    # Create figure
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(N_TRAVERSALS, 10, figure=fig, hspace=0.4, wspace=0.8)

    # Panel A: Latent traversals
    print("\nGenerating latent traversals...")

    # Use a random sample as base
    base_idx = np.random.randint(0, len(latent_codes))
    base_z = torch.FloatTensor(latent_codes[base_idx:base_idx+1])

    for row, dim_idx in enumerate(tqdm(top_dims, desc="Traversals")):
        # Traverse from -3 to +3 standard deviations
        std = np.std(latent_codes[:, dim_idx])
        values = np.linspace(-3*std, 3*std, 7)

        for col, val in enumerate(values):
            ax = fig.add_subplot(gs[row, col])

            # Modify latent code
            z_mod = base_z.clone()
            z_mod[0, dim_idx] = float(val)

            # Decode
            with torch.no_grad():
                decoded, max_vals = model.decode(z_mod.to(DEVICE))

            # Unnormalize
            curve_normalized = decoded[0].cpu().numpy()
            max_vals_np = max_vals[0].cpu().numpy()
            curve = curve_normalized * max_vals_np[:, np.newaxis]

            # Plot
            for species in range(curve.shape[0]):
                ax.plot(curve[species], alpha=0.7, linewidth=1.5)

            ax.set_ylim(-0.05, 1.1)
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(f'Dim {dim_idx}\n(KL={kl_per_dim[dim_idx]:.2f})',
                            fontsize=9, fontweight='bold')

            if row == 0:
                ax.set_title(f'{val/std:.1f}σ', fontsize=9)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Panel B: Correlation heatmap
    ax_corr = fig.add_subplot(gs[:4, 7:10])

    im = ax_corr.imshow(correlations.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax_corr.set_xlabel('Latent Dimension', fontweight='bold')
    ax_corr.set_ylabel('Feature', fontweight='bold')
    ax_corr.set_title('Latent-Feature Correlations', fontweight='bold', fontsize=13)
    ax_corr.set_yticks(range(len(feature_names)))
    ax_corr.set_yticklabels(feature_names, fontsize=8)

    plt.colorbar(im, ax=ax_corr, label='Pearson r')

    # Panel C: SAP scores
    ax_sap = fig.add_subplot(gs[4:6, 7:10])

    colors = plt.cm.viridis(np.array(sap_scores) / max(sap_scores))
    bars = ax_sap.barh(range(len(sap_scores)), sap_scores, color=colors, edgecolor='black')
    ax_sap.set_yticks(range(len(feature_names)))
    ax_sap.set_yticklabels(feature_names, fontsize=9)
    ax_sap.set_xlabel('SAP Score', fontweight='bold')
    ax_sap.set_title('Separated Attribute Predictability', fontweight='bold', fontsize=13)
    ax_sap.grid(alpha=0.2, axis='x')

    # Panel D: Top associations
    ax_assoc = fig.add_subplot(gs[6:, 7:10])

    # Find top 10 absolute correlations
    abs_corr = np.abs(correlations)
    top_indices = np.unravel_index(np.argsort(abs_corr.ravel())[::-1][:10], abs_corr.shape)

    text_str = "Top 10 Latent-Feature Associations:\n" + "="*45 + "\n"
    for i, (dim, feat_idx) in enumerate(zip(top_indices[0], top_indices[1]), 1):
        corr_val = correlations[dim, feat_idx]
        text_str += f"{i:2d}. Dim {dim:2d} ↔ {feature_names[feat_idx]:20s}: r={corr_val:+.3f}\n"

    ax_assoc.text(0.05, 0.95, text_str, transform=ax_assoc.transAxes,
                 fontsize=9, family='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax_assoc.axis('off')

    plt.savefig(OUTPUT_DIR / 'Figure2_Disentanglement.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure2_Disentanglement.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 2")
    plt.close()


# ============================================================================
# FIGURE 3: GENERATIVE QUALITY
# ============================================================================

def generate_figure3_generative_quality(data):
    """
    Comprehensive generative quality assessment.

    Row 1: Reconstruction examples (original vs reconstructed)
    Row 2: Interpolation examples (start and end points)
    Row 3: Random samples from prior
    Row 4: Attribute manipulation examples
    """
    print("\n" + "="*70)
    print("FIGURE 3: GENERATIVE QUALITY")
    print("="*70)

    model = data['model']
    curves = data['curves']
    reconstructions = data['reconstructions']
    latent_codes = data['latent_codes']
    features = data['features']

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: Reconstructions
    print("\nGenerating reconstruction examples...")
    recon_indices = np.random.choice(len(curves), 5, replace=False)

    for col, idx in enumerate(recon_indices):
        ax = fig.add_subplot(gs[0, col])

        original = curves[idx]
        recon = reconstructions[idx]

        # Trim to minimum length
        min_len = min(original.shape[1], recon.shape[1])
        original_trim = original[:, :min_len]
        recon_trim = recon[:, :min_len]

        # Show only 3 species for clarity
        for species in range(3):
            # Original: solid line
            ax.plot(original_trim[species], alpha=0.7, linewidth=2.5,
                   color=f'C{species}', linestyle='-',
                   label=f'Sp{species}' if col == 0 else '')
            # Reconstruction: dashed line with markers
            ax.plot(recon_trim[species], alpha=0.9, linewidth=1.5, linestyle='--',
                   color=f'C{species}', marker='o', markersize=2, markevery=10,
                   label=f'Rec{species}' if col == 0 else '')

        mse = np.mean((original_trim - recon_trim)**2)
        ax.set_title(f'Reconstruction {col+1}\nMSE={mse:.4f}', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Abundance' if col == 0 else '')
        ax.grid(alpha=0.2)

        if col == 0:
            ax.legend(fontsize=8, loc='upper right', ncol=2)

    # Row 2: Interpolations (ONE path from A to B shown across 5 columns)
    print("\nGenerating interpolation examples...")

    # Pick TWO samples that are FAR APART in latent space
    distances_matrix = np.linalg.norm(latent_codes[:, None, :] - latent_codes[None, :, :], axis=2)
    np.fill_diagonal(distances_matrix, 0)
    idx1, idx2 = np.unravel_index(distances_matrix.argmax(), distances_matrix.shape)

    z_A = torch.FloatTensor(latent_codes[idx1:idx1+1])
    z_B = torch.FloatTensor(latent_codes[idx2:idx2+1])
    original_A = curves[idx1]
    original_B = curves[idx2]
    recon_A = reconstructions[idx1]
    recon_B = reconstructions[idx2]

    # Column layout: Decode A, interp1, interp2, interp3, Decode B
    # Use decoded versions of A and B for fair comparison
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ['A (decoded)', 'A→B (25%)', 'A→B (50%)', 'A→B (75%)', 'B (decoded)']

    for col, (alpha, label) in enumerate(zip(alphas, labels)):
        ax = fig.add_subplot(gs[1, col])

        # Always decode from latent space for consistent comparison
        z_interp = (1 - alpha) * z_A + alpha * z_B

        with torch.no_grad():
            decoded, max_vals = model.decode(z_interp.to(DEVICE))

        curve_normalized = decoded[0].cpu().numpy()
        max_vals_np = max_vals[0].cpu().numpy()
        curve = curve_normalized * max_vals_np[:, np.newaxis]

        for species in range(3):
            ax.plot(curve[species], alpha=0.8, linewidth=2.5, color=f'C{species}')

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Abundance' if col == 0 else '')
        ax.grid(alpha=0.2)

    # Row 3: Random samples
    print("\nGenerating random samples...")

    for col in range(5):
        ax = fig.add_subplot(gs[2, col])

        # Sample from prior
        z_random = torch.randn(1, model_config['latent_dim'])

        with torch.no_grad():
            decoded, max_vals = model.decode(z_random.to(DEVICE))

        # Unnormalize
        curve_normalized = decoded[0].cpu().numpy()
        max_vals_np = max_vals[0].cpu().numpy()
        curve = curve_normalized * max_vals_np[:, np.newaxis]

        for species in range(3):
            ax.plot(curve[species], alpha=0.8, linewidth=2.5, color=f'C{species}')

        ax.set_title(f'Random Sample {col+1}', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Abundance' if col == 0 else '')
        ax.grid(alpha=0.2)

    # Row 4: Multi-attribute manipulation (show control over different features)
    print("\nGenerating multi-attribute manipulation examples...")

    # Find dimensions most correlated with different features
    feature_names_for_control = ['diversity', 'temporal_variance', 'steady_state_sum', 'n_survivors', 'overshoot_mean']
    controlled_dims = []

    for feat_name in feature_names_for_control:
        corrs = [np.corrcoef(latent_codes[:, i], features[feat_name])[0, 1]
                for i in range(latent_codes.shape[1])]
        best_dim = np.argmax(np.abs(corrs))
        controlled_dims.append((best_dim, feat_name))

    # Pick a neutral base sample (near median for key scalar features)
    scalar_features = ['diversity', 'temporal_variance', 'steady_state_sum', 'n_survivors']
    median_features = {k: np.median(features[k]) for k in scalar_features}
    distances_to_median = np.sum([(features[k] - median_features[k])**2 for k in scalar_features], axis=0)
    base_idx = np.argmin(distances_to_median)
    base_z = torch.FloatTensor(latent_codes[base_idx:base_idx+1])

    # Show manipulating each attribute with STRONG modification for visibility
    # Alternate between increasing/decreasing to show variety
    modifications = [3, -3, 3, -3, 3]  # in units of σ

    for col, ((dim_idx, feat_name), mod_sign) in enumerate(zip(controlled_dims, modifications)):
        ax = fig.add_subplot(gs[3, col])

        # Modify this dimension strongly
        z_mod = base_z.clone()
        std = np.std(latent_codes[:, dim_idx])
        mean = np.mean(latent_codes[:, dim_idx])
        # Set to extreme value (mean ± 3σ)
        z_mod[0, dim_idx] = float(mean + mod_sign * std)

        with torch.no_grad():
            decoded, max_vals = model.decode(z_mod.to(DEVICE))

        # Unnormalize
        curve_normalized = decoded[0].cpu().numpy()
        max_vals_np = max_vals[0].cpu().numpy()
        curve = curve_normalized * max_vals_np[:, np.newaxis]

        for species in range(3):
            ax.plot(curve[species], alpha=0.8, linewidth=2.5, color=f'C{species}')

        # Clean up feature name
        display_name = feat_name.replace('_', ' ').title()
        arrow = '↑' if mod_sign > 0 else '↓'
        ax.set_title(f'{arrow} {display_name}', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Abundance' if col == 0 else '')
        ax.grid(alpha=0.2)

    plt.savefig(OUTPUT_DIR / 'Figure3_Generative_Quality.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure3_Generative_Quality.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 3")
    plt.close()


# ============================================================================
# FIGURE 4: MANIFOLD TOPOLOGY
# ============================================================================

def generate_figure4_manifold_topology(data):
    """
    Manifold structure and topology analysis.

    Panel A: PCA variance explained
    Panel B: t-SNE embedding
    Panel C: Local neighborhood structure
    Panel D: Geodesic interpolations
    """
    print("\n" + "="*70)
    print("FIGURE 4: MANIFOLD TOPOLOGY")
    print("="*70)

    model = data['model']
    latent_codes = data['latent_codes']
    features = data['features']

    # Subsample if needed
    if len(latent_codes) > N_SAMPLES:
        idx = np.random.choice(len(latent_codes), N_SAMPLES, replace=False)
        latent_subset = latent_codes[idx]
        features_subset = {k: v[idx] for k, v in features.items()}
    else:
        latent_subset = latent_codes
        features_subset = features

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: PCA variance
    print("\nComputing PCA...")
    pca = PCA()
    pca.fit(latent_subset)

    ax_pca = fig.add_subplot(gs[0, 0])

    variance_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(variance_ratio)

    # Show all 30 dimensions
    n_dims = len(variance_ratio)
    ax_pca.bar(range(n_dims), variance_ratio,
              alpha=0.7, color='steelblue', edgecolor='black')
    ax_pca.plot(range(n_dims), cumsum, 'ro-', linewidth=2)

    ax_pca.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, label='90%')
    ax_pca.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')

    ax_pca.set_xlabel('Principal Component', fontweight='bold')
    ax_pca.set_ylabel('Variance Explained', fontweight='bold')
    ax_pca.set_title('PCA Variance Explained', fontweight='bold', fontsize=12)
    ax_pca.legend()
    ax_pca.grid(alpha=0.2)

    # Panel B: t-SNE
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    embedding_tsne = tsne.fit_transform(latent_subset)

    ax_tsne = fig.add_subplot(gs[0, 1])

    scatter = ax_tsne.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1],
                             c=features_subset['diversity'], cmap='viridis',
                             s=15, alpha=0.6, edgecolors='none')

    ax_tsne.set_xlabel('t-SNE 1', fontweight='bold')
    ax_tsne.set_ylabel('t-SNE 2', fontweight='bold')
    ax_tsne.set_title('t-SNE Embedding (by Diversity)', fontweight='bold', fontsize=12)
    ax_tsne.grid(alpha=0.2)

    plt.colorbar(scatter, ax=ax_tsne, label='Diversity')

    # Panel C: Local neighborhood structure
    print("\nAnalyzing local neighborhoods...")
    ax_neigh = fig.add_subplot(gs[0, 2:])

    # Pick a random point and show its k nearest neighbors
    query_idx = np.random.randint(0, len(latent_subset))
    query_z = latent_subset[query_idx]

    # Compute distances
    distances = np.linalg.norm(latent_subset - query_z, axis=1)
    nearest_indices = np.argsort(distances)[:6]  # Query + 5 neighbors

    # Decode all
    for i, idx in enumerate(nearest_indices):
        ax_sub = ax_neigh

        z = torch.FloatTensor(latent_subset[idx:idx+1])
        with torch.no_grad():
            decoded, max_vals = model.decode(z.to(DEVICE))

        # Unnormalize the curve
        curve_normalized = decoded[0].cpu().numpy()
        max_vals_np = max_vals[0].cpu().numpy()
        curve = curve_normalized * max_vals_np[:, np.newaxis]

        # Offset each curve vertically
        offset = i * 1.3

        for species in range(min(3, curve.shape[0])):
            alpha = 1.0 if i == 0 else 0.5
            linewidth = 3 if i == 0 else 2
            ax_neigh.plot(curve[species] + offset, alpha=alpha,
                         linewidth=linewidth, color=f'C{species}')

        # Label
        label = f"Query (Div={features_subset['diversity'][idx]:.2f})" if i == 0 else f"Neighbor {i} (d={distances[idx]:.2f})"
        ax_neigh.text(-5, offset + 0.5, label, fontsize=9, verticalalignment='center')

    ax_neigh.set_xlabel('Time', fontweight='bold')
    ax_neigh.set_ylabel('Abundance (offset)', fontweight='bold')
    ax_neigh.set_title('Local Neighborhood Structure', fontweight='bold', fontsize=12)
    ax_neigh.set_ylim(-0.5, (len(nearest_indices) * 1.3) + 0.5)
    ax_neigh.spines['right'].set_visible(False)
    ax_neigh.spines['top'].set_visible(False)

    # Panel D: Latent arithmetic examples
    print("\nGenerating latent arithmetic examples...")

    # z1 + (z2 - z3) style arithmetic
    idx1, idx2, idx3 = np.random.choice(len(latent_subset), 3, replace=False)
    z1 = torch.FloatTensor(latent_subset[idx1:idx1+1])
    z2 = torch.FloatTensor(latent_subset[idx2:idx2+1])
    z3 = torch.FloatTensor(latent_subset[idx3:idx3+1])

    examples = [
        (z1, None, None, 1.0, f"Sample A\n(z₁)"),
        (z2, None, None, 1.0, f"Sample B\n(z₂)"),
        (z3, None, None, 1.0, f"Sample C\n(z₃)"),
        (z1, z2, z3, 1.0, "Arithmetic\nz₁ + (z₂ - z₃)"),
    ]

    for col, (za, zb, zc, alpha, label) in enumerate(examples):
        ax = fig.add_subplot(gs[1, col])

        decoded, max_vals, _ = latent_arithmetic(model, za, zb, zc, alpha)
        # Unnormalize the curve
        curve_normalized = decoded[0]  # Shape: (7, timesteps)
        curve_unnormalized = curve_normalized * max_vals[0][:, np.newaxis]  # Apply scaling

        # Plot first 3 species with unnormalized values
        for species in range(3):
            ax.plot(curve_unnormalized[species], alpha=0.8, linewidth=2.5, color=f'C{species}')

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel('Abundance' if col == 0 else '')
        ax.set_xlabel('Time')
        ax.grid(alpha=0.2)

    plt.savefig(OUTPUT_DIR / 'Figure4_Manifold_Topology.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure4_Manifold_Topology.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 4")
    plt.close()


# ============================================================================
# FIGURE 5: CAPACITY UTILIZATION
# ============================================================================

def generate_figure5_capacity_utilization(data):
    """
    Model capacity and information content analysis.

    Panel A: Active units (KL > threshold)
    Panel B: Information bottleneck analysis
    Panel C: Dimension ablation study
    Panel D: Cumulative information content
    """
    print("\n" + "="*70)
    print("FIGURE 5: CAPACITY UTILIZATION")
    print("="*70)

    latent_mu = data['latent_mu']
    latent_log_var = data['latent_log_var']
    latent_codes = data['latent_codes']
    curves = data['curves']
    reconstructions = data['reconstructions']

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Compute KL per dimension
    kl_per_dim = compute_kl_per_dimension(latent_mu, latent_log_var)

    # Panel A: Active units at different thresholds
    ax_active = fig.add_subplot(gs[0, 0])

    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    active_counts = [np.sum(kl_per_dim > t) for t in thresholds]

    bars = ax_active.bar(range(len(thresholds)), active_counts,
                        color='steelblue', edgecolor='black', alpha=0.7)
    ax_active.set_xticks(range(len(thresholds)))
    ax_active.set_xticklabels([f'{t:.3f}' for t in thresholds])
    ax_active.set_xlabel('KL Threshold', fontweight='bold')
    ax_active.set_ylabel('Number of Active Dimensions', fontweight='bold')
    ax_active.set_title('Active Units at Different Thresholds', fontweight='bold', fontsize=13)
    ax_active.grid(alpha=0.2, axis='y')

    # Add percentages on bars
    for i, (bar, count) in enumerate(zip(bars, active_counts)):
        height = bar.get_height()
        pct = 100 * count / len(kl_per_dim)
        ax_active.text(bar.get_x() + bar.get_width()/2., height,
                      f'{count}\n({pct:.0f}%)',
                      ha='center', va='bottom', fontsize=9)

    # Panel B: Information bottleneck
    ax_info = fig.add_subplot(gs[0, 1])

    # Sort dimensions by KL
    sorted_indices = np.argsort(kl_per_dim)[::-1]
    sorted_kl = kl_per_dim[sorted_indices]
    cumsum_kl = np.cumsum(sorted_kl)
    cumsum_kl_norm = cumsum_kl / cumsum_kl[-1]

    ax_info.plot(range(len(sorted_kl)), sorted_kl, 'o-',
                color='steelblue', linewidth=2, markersize=4, label='Individual KL')
    ax_info.set_xlabel('Dimension (sorted by KL)', fontweight='bold')
    ax_info.set_ylabel('KL Divergence', fontweight='bold', color='steelblue')
    ax_info.tick_params(axis='y', labelcolor='steelblue')
    ax_info.grid(alpha=0.2)

    # Cumulative on secondary axis
    ax_info2 = ax_info.twinx()
    ax_info2.plot(range(len(cumsum_kl_norm)), cumsum_kl_norm, 's-',
                 color='coral', linewidth=2, markersize=4, label='Cumulative')
    ax_info2.set_ylabel('Cumulative Information (normalized)', fontweight='bold', color='coral')
    ax_info2.tick_params(axis='y', labelcolor='coral')
    ax_info2.axhline(y=0.90, color='green', linestyle='--', alpha=0.5)
    ax_info2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5)

    # Find dimensions needed for 90% and 95%
    dims_90 = np.argmax(cumsum_kl_norm >= 0.90) + 1
    dims_95 = np.argmax(cumsum_kl_norm >= 0.95) + 1

    ax_info.set_title(f'Information Bottleneck\n(90%: {dims_90} dims, 95%: {dims_95} dims)',
                     fontweight='bold', fontsize=13)

    # Panel C: Reconstruction quality metrics
    ax_recon = fig.add_subplot(gs[1, 0])

    # Compute MSE per sample (trim to minimum length)
    min_len = min(curves.shape[2], reconstructions.shape[2])
    curves_trim = curves[:, :, :min_len]
    recons_trim = reconstructions[:, :, :min_len]
    mse_per_sample = np.mean((curves_trim - recons_trim)**2, axis=(1, 2))

    # Histogram
    ax_recon.hist(mse_per_sample, bins=50, color='steelblue',
                 alpha=0.7, edgecolor='black')
    ax_recon.axvline(x=np.median(mse_per_sample), color='red',
                    linestyle='--', linewidth=2, label=f'Median: {np.median(mse_per_sample):.4f}')
    ax_recon.axvline(x=np.mean(mse_per_sample), color='orange',
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(mse_per_sample):.4f}')

    ax_recon.set_xlabel('Reconstruction MSE', fontweight='bold')
    ax_recon.set_ylabel('Count', fontweight='bold')
    ax_recon.set_title('Reconstruction Quality Distribution', fontweight='bold', fontsize=13)
    ax_recon.legend()
    ax_recon.grid(alpha=0.2, axis='y')

    # Panel D: Variance captured by latent dimensions
    ax_var = fig.add_subplot(gs[1, 1])

    # Compute variance of each latent dimension
    latent_vars = np.var(latent_codes, axis=0)
    sorted_var_indices = np.argsort(latent_vars)[::-1]
    sorted_vars = latent_vars[sorted_var_indices]

    colors = plt.cm.viridis(sorted_vars / sorted_vars.max())
    bars = ax_var.bar(range(len(sorted_vars)), sorted_vars, color=colors, edgecolor='black')

    ax_var.set_xlabel('Dimension (sorted by variance)', fontweight='bold')
    ax_var.set_ylabel('Variance', fontweight='bold')
    ax_var.set_title('Latent Dimension Variance', fontweight='bold', fontsize=13)
    ax_var.grid(alpha=0.2, axis='y')

    # Add stats
    textstr = f'Mean variance: {np.mean(latent_vars):.3f}\n'
    textstr += f'Std variance: {np.std(latent_vars):.3f}\n'
    textstr += f'Max/Min ratio: {sorted_vars[0]/sorted_vars[-1]:.1f}x'
    ax_var.text(0.98, 0.97, textstr, transform=ax_var.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10)

    plt.savefig(OUTPUT_DIR / 'Figure5_Capacity_Utilization.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure5_Capacity_Utilization.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 5")
    plt.close()


# ============================================================================
# FIGURE 6: DYNAMICS ENCODING
# ============================================================================

def generate_figure6_dynamics_encoding(data):
    """
    How different dynamical features are encoded in latent space.

    Panel A-F: Scatter plots showing latent dims vs key dynamical features
    Panel G: Feature reconstruction from latent codes (linear decodability)
    """
    print("\n" + "="*70)
    print("FIGURE 6: DYNAMICS ENCODING")
    print("="*70)

    latent_codes = data['latent_codes']
    features = data['features']

    # Select key features
    key_features = [
        ('peak_time_mean', 'Mean Peak Time'),
        ('steady_state_sum', 'Total Steady State'),
        ('overshoot_mean', 'Mean Overshoot'),
        ('diversity', 'Shannon Diversity'),
        ('n_survivors', 'Number of Survivors'),
        ('dominance', 'Dominance Index'),
    ]

    # Find best latent dimension for each feature
    print("\nFinding best latent dimensions for each feature...")
    best_dims = []
    best_corrs = []

    for feat_name, _ in key_features:
        correlations = [abs(np.corrcoef(latent_codes[:, i], features[feat_name])[0, 1])
                       for i in range(latent_codes.shape[1])]
        best_dim = np.argmax(correlations)
        best_dims.append(best_dim)
        best_corrs.append(correlations[best_dim])

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Panels A-F: Scatter plots
    for i, ((feat_name, feat_label), dim, corr) in enumerate(zip(key_features, best_dims, best_corrs)):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        # Subsample for clarity
        if len(latent_codes) > 5000:
            idx = np.random.choice(len(latent_codes), 5000, replace=False)
            x = latent_codes[idx, dim]
            y = features[feat_name][idx]
        else:
            x = latent_codes[:, dim]
            y = features[feat_name]

        # Scatter with density coloring
        ax.hexbin(x, y, gridsize=30, cmap='Blues', mincnt=1)

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel(f'Latent Dim {dim}', fontweight='bold')
        ax.set_ylabel(feat_label, fontweight='bold')
        ax.set_title(f'{feat_label}\n|r| = {corr:.3f}', fontweight='bold', fontsize=11)
        ax.grid(alpha=0.2)

    # Panel G: Linear decodability analysis
    ax_decode = fig.add_subplot(gs[:, 3])

    print("\nComputing linear decodability...")

    # Test linear regression for all features
    feature_list = ['peak_time_mean', 'steady_state_sum', 'overshoot_mean',
                   'diversity', 'n_survivors', 'dominance', 'temporal_variance']

    r2_scores = []

    scaler = StandardScaler()
    X = scaler.fit_transform(latent_codes)

    for feat_name in feature_list:
        y = features[feat_name]

        # Train/test split
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Fit linear model
        model = LinearRegression()
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        r2_scores.append(r2)

    # Bar plot
    colors = plt.cm.viridis(np.array(r2_scores))
    bars = ax_decode.barh(range(len(feature_list)), r2_scores,
                          color=colors, edgecolor='black', alpha=0.8)

    ax_decode.set_yticks(range(len(feature_list)))
    ax_decode.set_yticklabels(feature_list, fontsize=10)
    ax_decode.set_xlabel('R² Score', fontweight='bold', fontsize=12)
    ax_decode.set_title('Linear Decodability\n(80/20 Train/Test Split)',
                       fontweight='bold', fontsize=13)
    ax_decode.set_xlim(0, 1)
    ax_decode.grid(alpha=0.2, axis='x')

    # Add values on bars
    for bar, score in zip(bars, r2_scores):
        width = bar.get_width()
        ax_decode.text(width, bar.get_y() + bar.get_height()/2.,
                      f' {score:.3f}',
                      ha='left', va='center', fontsize=9, fontweight='bold')

    # Add average line
    avg_r2 = np.mean(r2_scores)
    ax_decode.axvline(x=avg_r2, color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {avg_r2:.3f}')
    ax_decode.legend(fontsize=10)

    plt.savefig(OUTPUT_DIR / 'Figure6_Dynamics_Encoding.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure6_Dynamics_Encoding.pdf', bbox_inches='tight')
    print(f"\n✓ Saved Figure 6")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("STATE-OF-THE-ART LATENT SPACE ANALYSIS")
    print("="*70)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Random seed: {RANDOM_SEED}")

    # Load and encode
    data = load_data_and_encode()

    # Generate all figures
    generate_figure1_latent_structure(data)
    generate_figure2_disentanglement(data)
    generate_figure3_generative_quality(data)
    generate_figure4_manifold_topology(data)
    generate_figure5_capacity_utilization(data)
    generate_figure6_dynamics_encoding(data)

    # Summary statistics
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY STATISTICS")
    print("="*70)

    kl_per_dim = compute_kl_per_dimension(data['latent_mu'], data['latent_log_var'])

    print(f"\nLatent Space Dimensionality:")
    print(f"  Total dimensions: {len(kl_per_dim)}")
    print(f"  Active (KL>0.01): {np.sum(kl_per_dim > 0.01)}")
    print(f"  Highly active (KL>0.1): {np.sum(kl_per_dim > 0.1)}")
    print(f"  Mean KL: {kl_per_dim.mean():.4f}")
    print(f"  Total KL: {kl_per_dim.sum():.4f}")

    # Compute MSE (trim to minimum length)
    min_len = min(data['curves'].shape[2], data['reconstructions'].shape[2])
    curves_trim = data['curves'][:, :, :min_len]
    recons_trim = data['reconstructions'][:, :, :min_len]

    mse = np.mean((curves_trim - recons_trim)**2)
    print(f"\nReconstruction Quality:")
    print(f"  Mean MSE: {mse:.6f}")
    print(f"  Median MSE: {np.median(np.mean((curves_trim - recons_trim)**2, axis=(1,2))):.6f}")

    print(f"\n✓ All figures saved to: {OUTPUT_DIR}/")
    print(f"\n{'='*70}")
    print("🎉 ANALYSIS COMPLETE! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()
