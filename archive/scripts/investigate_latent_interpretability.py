"""
Deep investigation of latent space interpretability.

What do the latent dimensions represent?

Hypotheses to test:
1. Eigenvalues/eigenvectors of interaction matrix
2. Dynamical properties (frequencies, damping, phase)
3. Species interaction patterns
4. Temporal characteristics (oscillation, trends)
5. Initial condition effects
6. Attractor properties

This is a rigorous analysis for journal publication.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

from src.models.cvae import LSTM_VAE
from src.utils.config import DEVICE

# Use 30D model
model_config = {
    "in_channels": 7,
    "n_curves": 7,
    "seq_len": 65,
    "latent_dim": 30,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "scale_prediction_mode": 'log',
    "use_scale_conditioning": True,
}

MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'
TEST_ROUTE = 'data/TEST_FINAL_PROCESSED.pkl'
OUTPUT_DIR = 'final figures/'

print("="*80)
print("LATENT SPACE INTERPRETABILITY INVESTIGATION")
print("What do the 30 latent dimensions represent?")
print("="*80)


def load_model_and_data():
    """Load model and data."""
    print("\nLoading model and data...")

    model = LSTM_VAE(config=model_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(TEST_ROUTE, 'rb') as f:
        data_package = pickle.load(f)

    print(f"✓ Loaded. Test samples: {len(data_package['data'])}")
    return model, data_package['data'], data_package['reconstruction_max_values']


def extract_dynamical_features(time_series):
    """
    Extract interpretable dynamical features from time series.

    Args:
        time_series: shape (n_species, seq_len)

    Returns:
        dict of features
    """
    n_species, seq_len = time_series.shape
    features = {}

    # 1. Oscillation characteristics (per species)
    oscillation_freqs = []
    oscillation_powers = []

    for i in range(n_species):
        signal_i = time_series[i, :]

        # FFT
        yf = fft(signal_i)
        xf = fftfreq(seq_len)[:seq_len//2]
        power = 2.0/seq_len * np.abs(yf[:seq_len//2])

        if len(power) > 1:
            dom_idx = np.argmax(power[1:]) + 1  # Skip DC
            oscillation_freqs.append(xf[dom_idx])
            oscillation_powers.append(power[dom_idx])
        else:
            oscillation_freqs.append(0)
            oscillation_powers.append(0)

    features['osc_freqs'] = np.array(oscillation_freqs)
    features['osc_powers'] = np.array(oscillation_powers)
    features['mean_osc_freq'] = np.mean(oscillation_freqs)
    features['std_osc_freq'] = np.std(oscillation_freqs)

    # 2. Trend characteristics
    trends = []
    for i in range(n_species):
        # Linear trend
        x = np.arange(seq_len)
        y = time_series[i, :]
        slope = np.polyfit(x, y, 1)[0]
        trends.append(slope)

    features['trends'] = np.array(trends)
    features['mean_trend'] = np.mean(trends)

    # 3. Variance characteristics
    variances = np.var(time_series, axis=1)
    features['variances'] = variances
    features['total_variance'] = np.sum(variances)
    features['max_variance'] = np.max(variances)

    # 4. Cross-correlation structure
    correlations = []
    for i in range(n_species):
        for j in range(i+1, n_species):
            corr = np.corrcoef(time_series[i, :], time_series[j, :])[0, 1]
            correlations.append(corr)

    features['mean_correlation'] = np.mean(correlations)
    features['std_correlation'] = np.std(correlations)
    features['max_correlation'] = np.max(np.abs(correlations))

    # 5. Curvature (second derivative)
    curvatures = []
    for i in range(n_species):
        second_deriv = np.diff(time_series[i, :], n=2)
        curvatures.append(np.mean(np.abs(second_deriv)))

    features['mean_curvature'] = np.mean(curvatures)

    # 6. Extrema count (peaks and troughs)
    extrema_counts = []
    for i in range(n_species):
        peaks, _ = signal.find_peaks(time_series[i, :])
        troughs, _ = signal.find_peaks(-time_series[i, :])
        extrema_counts.append(len(peaks) + len(troughs))

    features['mean_extrema'] = np.mean(extrema_counts)

    # 7. Phase relationships (pairwise)
    phase_diffs = []
    for i in range(n_species):
        for j in range(i+1, n_species):
            # Cross-correlation to find phase shift
            xcorr = np.correlate(time_series[i, :], time_series[j, :], mode='full')
            lag = np.argmax(xcorr) - (seq_len - 1)
            phase_diffs.append(lag)

    features['mean_phase_diff'] = np.mean(np.abs(phase_diffs))

    # 8. Synchronization measure
    # How similar are the shapes across species?
    sync_scores = []
    for i in range(n_species):
        for j in range(i+1, n_species):
            # Normalized cross-correlation
            corr = np.corrcoef(time_series[i, :], time_series[j, :])[0, 1]
            sync_scores.append(corr)

    features['synchronization'] = np.mean(sync_scores)

    # 9. Stability proxy (damping)
    # If oscillations decay over time
    dampings = []
    for i in range(n_species):
        first_half_var = np.var(time_series[i, :seq_len//2])
        second_half_var = np.var(time_series[i, seq_len//2:])
        if first_half_var > 0:
            damping = (first_half_var - second_half_var) / first_half_var
        else:
            damping = 0
        dampings.append(damping)

    features['mean_damping'] = np.mean(dampings)

    return features


def encode_and_extract_features(model, test_data, test_max_vals, n_samples=3000):
    """
    Encode samples and extract both latent codes and dynamical features.
    """
    print(f"\nEncoding {n_samples} samples and extracting features...")

    latent_codes = []
    all_features = []
    original_data = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Processing"):
            # Encode to latent
            x_norm = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            max_vals = torch.tensor(test_max_vals[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            x_rnn = x_norm.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(x_rnn)
            encoded_summary = h_n.permute(1, 0, 2).contiguous().view(1, -1)

            z_shape = model.shape_projector(encoded_summary)
            z_scale = model.scale_encoder(max_vals)
            bottleneck_input = torch.cat([z_shape, z_scale], dim=1)

            mu = model.fc_mu(bottleneck_input)

            latent_codes.append(mu.cpu().numpy()[0])

            # Extract features from original scale data
            data_norm = test_data[i]
            max_vals_np = test_max_vals[i]
            data_original = data_norm * max_vals_np.reshape(1, -1, 1)

            features = extract_dynamical_features(data_original[0])
            all_features.append(features)
            original_data.append(data_original[0])

    latent_codes = np.array(latent_codes)

    # Convert features dict list to structured arrays
    feature_matrix = {}
    feature_names = all_features[0].keys()

    for fname in feature_names:
        values = []
        for f in all_features:
            val = f[fname]
            if isinstance(val, np.ndarray):
                values.append(val)
            else:
                values.append(val)

        values = np.array(values)
        if len(values.shape) == 1:
            feature_matrix[fname] = values
        else:
            # For array features, add each component
            for j in range(values.shape[1]):
                feature_matrix[f'{fname}_{j}'] = values[:, j]

    print(f"✓ Extracted {len(feature_matrix)} feature dimensions")

    return latent_codes, feature_matrix, original_data


def analyze_latent_feature_correlations(latent_codes, feature_matrix):
    """
    Analyze correlations between latent dimensions and dynamical features.
    """
    print("\n" + "="*80)
    print("CORRELATING LATENT DIMENSIONS WITH DYNAMICAL FEATURES")
    print("="*80)

    # For each latent dimension, find strongest correlations
    n_latent = latent_codes.shape[1]

    results = []

    for latent_idx in range(n_latent):
        latent_dim = latent_codes[:, latent_idx]

        correlations = {}
        for feat_name, feat_values in feature_matrix.items():
            if len(feat_values) == len(latent_dim):
                # Pearson correlation
                r, p = pearsonr(latent_dim, feat_values)
                correlations[feat_name] = {'r': r, 'p': p, 'abs_r': abs(r)}

        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(),
                            key=lambda x: x[1]['abs_r'],
                            reverse=True)

        results.append({
            'latent_idx': latent_idx,
            'top_correlations': sorted_corrs[:5]
        })

    # Print summary
    print("\nTop correlations for each latent dimension:")
    print("-" * 80)

    for res in results[:10]:  # Show first 10
        latent_idx = res['latent_idx']
        print(f"\nLatent Dim {latent_idx}:")
        for feat_name, corr_info in res['top_correlations'][:3]:
            r = corr_info['r']
            p = corr_info['p']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat_name:30s}: r={r:+.3f} {sig}")

    return results


def find_interpretable_components(latent_codes, feature_matrix):
    """
    Use PCA on latent codes and try to interpret principal components.
    """
    print("\n" + "="*80)
    print("INTERPRETING PRINCIPAL COMPONENTS")
    print("="*80)

    # PCA on latent codes
    pca = PCA()
    latent_pca = pca.fit_transform(latent_codes)

    print(f"\nPCA Variance explained:")
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        print(f"  PC{i}: {pca.explained_variance_ratio_[i]*100:.2f}%")

    # For each PC, correlate with features
    print("\n" + "-"*80)
    print("Correlating PCs with dynamical features:")
    print("-"*80)

    pc_interpretations = []

    for pc_idx in range(min(10, latent_pca.shape[1])):
        pc = latent_pca[:, pc_idx]

        correlations = {}
        for feat_name, feat_values in feature_matrix.items():
            if len(feat_values) == len(pc):
                r, p = pearsonr(pc, feat_values)
                if abs(r) > 0.1:  # Only keep meaningful correlations
                    correlations[feat_name] = {'r': r, 'p': p}

        sorted_corrs = sorted(correlations.items(),
                            key=lambda x: abs(x[1]['r']),
                            reverse=True)

        pc_interpretations.append({
            'pc_idx': pc_idx,
            'variance_explained': pca.explained_variance_ratio_[pc_idx],
            'top_correlations': sorted_corrs[:5]
        })

        # Print
        print(f"\nPC{pc_idx} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance):")
        for feat_name, corr_info in sorted_corrs[:5]:
            r = corr_info['r']
            p = corr_info['p']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat_name:30s}: r={r:+.3f} {sig}")

    return pca, pc_interpretations


def test_predictability(latent_codes, feature_matrix):
    """
    Test if latent codes can predict dynamical features (and vice versa).
    """
    print("\n" + "="*80)
    print("TESTING PREDICTIVE RELATIONSHIPS")
    print("="*80)

    # Can latent codes predict features?
    print("\nCan latent codes predict dynamical features?")
    print("-" * 80)

    feature_predictions = {}

    # Test a few key features
    key_features = ['mean_osc_freq', 'total_variance', 'mean_correlation',
                   'synchronization', 'mean_damping', 'mean_extrema']

    for feat_name in key_features:
        if feat_name in feature_matrix:
            feat_values = feature_matrix[feat_name]

            # Ridge regression (with regularization)
            model = Ridge(alpha=1.0)

            # Train-test split
            n_train = int(0.8 * len(latent_codes))
            X_train = latent_codes[:n_train]
            y_train = feat_values[:n_train]
            X_test = latent_codes[n_train:]
            y_test = feat_values[n_train:]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # R² score
            ss_res = np.sum((y_test - y_pred)**2)
            ss_tot = np.sum((y_test - np.mean(y_test))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            feature_predictions[feat_name] = {
                'r2': r2,
                'coefficients': model.coef_
            }

            print(f"  {feat_name:25s}: R² = {r2:.3f}")

    return feature_predictions


def analyze_latent_dimension_roles(latent_codes, feature_matrix, correlation_results):
    """
    Synthesize results to assign interpretable roles to latent dimensions.
    """
    print("\n" + "="*80)
    print("ASSIGNING INTERPRETABLE ROLES TO LATENT DIMENSIONS")
    print("="*80)

    # Categorize dimensions by their strongest correlations
    dimension_roles = {}

    for res in correlation_results:
        latent_idx = res['latent_idx']
        top_corrs = res['top_correlations']

        if len(top_corrs) > 0:
            top_feat, top_info = top_corrs[0]
            abs_r = top_info['abs_r']

            # Categorize by feature type
            if 'osc' in top_feat or 'freq' in top_feat:
                category = 'Oscillation'
            elif 'trend' in top_feat or 'damping' in top_feat:
                category = 'Trend/Stability'
            elif 'variance' in top_feat or 'extrema' in top_feat:
                category = 'Amplitude/Variability'
            elif 'correlation' in top_feat or 'sync' in top_feat or 'phase' in top_feat:
                category = 'Species Interactions'
            elif 'curvature' in top_feat:
                category = 'Trajectory Shape'
            else:
                category = 'Mixed/Other'

            dimension_roles[latent_idx] = {
                'category': category,
                'top_feature': top_feat,
                'correlation': top_info['r'],
                'strength': abs_r
            }

    # Count categories
    category_counts = {}
    for role in dimension_roles.values():
        cat = role['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nLatent dimension categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:30s}: {count} dimensions")

    # Print strongest dimensions in each category
    print("\n" + "-"*80)
    print("Strongest dimensions by category:")
    print("-"*80)

    for category in category_counts.keys():
        dims_in_cat = [(idx, info) for idx, info in dimension_roles.items()
                      if info['category'] == category]
        dims_sorted = sorted(dims_in_cat, key=lambda x: x[1]['strength'], reverse=True)

        print(f"\n{category}:")
        for idx, info in dims_sorted[:3]:
            print(f"  Dim {idx:2d}: {info['top_feature']:25s} (r={info['correlation']:+.3f})")

    return dimension_roles


def create_interpretation_figure(pca, pc_interpretations, dimension_roles):
    """
    Create comprehensive visualization of latent space interpretation.
    """
    print("\n" + "="*80)
    print("GENERATING INTERPRETATION FIGURE")
    print("="*80)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel 1: PCA variance explained with interpretation
    ax1 = fig.add_subplot(gs[0, :2])

    var_explained = pca.explained_variance_ratio_[:20] * 100
    x_pos = np.arange(len(var_explained))

    bars = ax1.bar(x_pos, var_explained, alpha=0.7, color='steelblue',
                   edgecolor='black', linewidth=0.8)

    # Color code by interpretation strength
    for i, (bar, pc_info) in enumerate(zip(bars, pc_interpretations[:20])):
        if len(pc_info['top_correlations']) > 0:
            top_r = abs(pc_info['top_correlations'][0][1]['r'])
            if top_r > 0.5:
                bar.set_color('darkgreen')
            elif top_r > 0.3:
                bar.set_color('orange')

    ax1.set_xlabel('Principal Component', fontweight='bold')
    ax1.set_ylabel('Variance Explained (%)', fontweight='bold')
    ax1.set_title('PCA Components (colored by interpretability)',
                 fontweight='bold', fontsize=13)
    ax1.grid(alpha=0.3, axis='y')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', label='Strong correlation (|r|>0.5)'),
        Patch(facecolor='orange', label='Moderate correlation (|r|>0.3)'),
        Patch(facecolor='steelblue', label='Weak correlation')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Panel 2: Dimension role distribution
    ax2 = fig.add_subplot(gs[0, 2])

    categories = {}
    for role in dimension_roles.values():
        cat = role['category']
        categories[cat] = categories.get(cat, 0) + 1

    cats = list(categories.keys())
    counts = [categories[c] for c in cats]

    colors_cat = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    ax2.pie(counts, labels=cats, autopct='%1.0f%%', startangle=90,
           colors=colors_cat)
    ax2.set_title('Latent Dimension Roles', fontweight='bold', fontsize=12)

    # Panel 3-5: Top PC interpretations (text)
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')

        if i < len(pc_interpretations):
            pc_info = pc_interpretations[i]
            pc_idx = pc_info['pc_idx']
            var_exp = pc_info['variance_explained'] * 100

            text = f"PC{pc_idx} ({var_exp:.1f}% variance)\n"
            text += "="*40 + "\n\n"
            text += "Top Correlations:\n"

            for feat_name, corr_info in pc_info['top_correlations'][:5]:
                r = corr_info['r']
                text += f"{feat_name:25s}: {r:+.3f}\n"

            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax.set_title(f'Principal Component {pc_idx}', fontweight='bold')

    # Panels 6-8: Dimension role details
    categories_list = list(categories.keys())
    for i in range(min(3, len(categories_list))):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')

        category = categories_list[i]
        dims_in_cat = [(idx, info) for idx, info in dimension_roles.items()
                      if info['category'] == category]
        dims_sorted = sorted(dims_in_cat, key=lambda x: x[1]['strength'], reverse=True)

        text = f"{category}\n"
        text += "="*40 + "\n\n"
        text += "Top Dimensions:\n"

        for idx, info in dims_sorted[:7]:
            r = info['correlation']
            feat = info['top_feature'][:20]
            text += f"Dim {idx:2d}: {feat:20s} {r:+.3f}\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax.set_title(category, fontweight='bold')

    plt.suptitle('Latent Space Interpretability Analysis (30D Model)',
                fontweight='bold', fontsize=16)

    output_path = f'{OUTPUT_DIR}fig_latent_interpretability_30.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def save_detailed_report(correlation_results, pc_interpretations, dimension_roles,
                        feature_predictions):
    """
    Save comprehensive text report of findings.
    """
    print("\n" + "="*80)
    print("SAVING DETAILED REPORT")
    print("="*80)

    report_path = f'{OUTPUT_DIR}latent_interpretability_report_30D.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LATENT SPACE INTERPRETABILITY ANALYSIS - 30D MODEL\n")
        f.write("="*80 + "\n\n")

        # Summary of dimension roles
        f.write("DIMENSION ROLE SUMMARY\n")
        f.write("-"*80 + "\n")

        categories = {}
        for role in dimension_roles.values():
            cat = role['category']
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{cat:30s}: {count} dimensions\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED DIMENSION ANALYSIS\n")
        f.write("="*80 + "\n\n")

        for res in correlation_results:
            latent_idx = res['latent_idx']
            f.write(f"\nLatent Dimension {latent_idx}:\n")
            f.write("-" * 40 + "\n")

            for feat_name, corr_info in res['top_correlations'][:5]:
                r = corr_info['r']
                p = corr_info['p']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                f.write(f"  {feat_name:30s}: r={r:+.4f} {sig}\n")

        f.write("\n\n" + "="*80 + "\n")
        f.write("PRINCIPAL COMPONENT INTERPRETATION\n")
        f.write("="*80 + "\n\n")

        for pc_info in pc_interpretations:
            pc_idx = pc_info['pc_idx']
            var_exp = pc_info['variance_explained'] * 100

            f.write(f"\nPC{pc_idx} ({var_exp:.2f}% variance):\n")
            f.write("-" * 40 + "\n")

            for feat_name, corr_info in pc_info['top_correlations']:
                r = corr_info['r']
                p = corr_info['p']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                f.write(f"  {feat_name:30s}: r={r:+.4f} {sig}\n")

        f.write("\n\n" + "="*80 + "\n")
        f.write("PREDICTIVE POWER OF LATENT CODE\n")
        f.write("="*80 + "\n\n")

        f.write("Can latent codes predict dynamical features?\n\n")

        for feat_name, pred_info in feature_predictions.items():
            r2 = pred_info['r2']
            f.write(f"  {feat_name:30s}: R² = {r2:.4f}\n")

    print(f"✓ Saved detailed report: {report_path}")


def main():
    """Main execution."""

    # Load data
    model, test_data, test_max_vals = load_model_and_data()

    # Encode and extract features
    latent_codes, feature_matrix, original_data = encode_and_extract_features(
        model, test_data, test_max_vals, n_samples=3000
    )

    # Analyze correlations
    correlation_results = analyze_latent_feature_correlations(latent_codes, feature_matrix)

    # Interpret PCs
    pca, pc_interpretations = find_interpretable_components(latent_codes, feature_matrix)

    # Test predictability
    feature_predictions = test_predictability(latent_codes, feature_matrix)

    # Assign roles
    dimension_roles = analyze_latent_dimension_roles(
        latent_codes, feature_matrix, correlation_results
    )

    # Create figure
    create_interpretation_figure(pca, pc_interpretations, dimension_roles)

    # Save report
    save_detailed_report(correlation_results, pc_interpretations,
                        dimension_roles, feature_predictions)

    print("\n" + "="*80)
    print("INTERPRETABILITY ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  • {OUTPUT_DIR}fig_latent_interpretability_30.png/pdf")
    print(f"  • {OUTPUT_DIR}latent_interpretability_report_30D.txt")
    print("="*80)


if __name__ == "__main__":
    main()
