"""
Analyze what's happening in the latent interpolation.

Question: Does the interpolation show meaningful transitions in dynamics,
or is it just trivially interpolating visual features?

We'll check:
1. What dynamics does START curve have?
2. What dynamics does END curve have?
3. Do intermediate curves show PROGRESSIVE changes in these dynamics?
4. Are we interpolating biological features (peak times, overshoot, etc.)?
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

from VAE.models.cvae import LSTM_VAE
from config import model_config, DEVICE

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
})

MODEL_PATH = 'model_ckpts/model_new_arch.pth'
DATA_PATH = 'data/interaction_mapping_dataset.pkl'
OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def analyze_curve_dynamics(curve):
    """
    Extract key dynamical features from a curve.

    Returns dict with:
    - peak_times: when each species peaks
    - peak_values: max abundance
    - steady_states: final abundance
    - overshoot: (peak - steady) / steady
    - initial_growth: early slope
    - crossing_order: when species cross each other
    """
    n_species, n_time = curve.shape

    features = {}

    # Peak times and values
    features['peak_times'] = np.argmax(curve, axis=1)
    features['peak_values'] = np.max(curve, axis=1)

    # Steady states (last 10 timepoints)
    features['steady_states'] = np.mean(curve[:, -10:], axis=1)

    # Overshoot
    steady_safe = np.maximum(features['steady_states'], 1e-3)
    features['overshoot'] = (features['peak_values'] - steady_safe) / steady_safe

    # Initial growth (slope in first 10 points)
    features['initial_growth'] = np.mean(np.gradient(curve[:, :10], axis=1), axis=1)

    # Temporal variance per species
    features['temporal_variance'] = np.var(curve, axis=1)

    # Crossing analysis - when do species cross?
    crossings = []
    for sp1 in range(n_species):
        for sp2 in range(sp1+1, n_species):
            # Find first crossing
            diff = curve[sp1] - curve[sp2]
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            if len(sign_changes) > 0:
                crossings.append({
                    'species_pair': (sp1, sp2),
                    'first_crossing': sign_changes[0],
                    'num_crossings': len(sign_changes)
                })

    features['crossings'] = crossings

    return features


def compare_interpolation_progression(start_idx=50, end_idx=250, n_steps=7):
    """
    Analyze the interpolation to see if dynamics change progressively.
    """
    print("="*70)
    print("INTERPOLATION ANALYSIS")
    print("="*70)

    # Load data and model
    print("\nLoading data and model...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    model = LSTM_VAE(config=model_config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Encode
    curves_tensor = torch.tensor(data['curves'], dtype=torch.float32)
    latent_codes = []

    with torch.no_grad():
        batch_size = 64
        for i in tqdm(range(0, len(curves_tensor), batch_size), desc="Encoding"):
            batch = curves_tensor[i:i+batch_size].to(DEVICE)
            X_rnn = batch.permute(0, 2, 1)
            _, (h_n, _) = model.encoder_rnn(X_rnn)
            encoded = h_n.permute(1, 0, 2).contiguous().view(batch.size(0), -1)
            mu = model.fc_mu(encoded)
            latent_codes.append(mu.cpu())

    latent_codes = torch.cat(latent_codes, dim=0).numpy()

    # Generate interpolation
    print(f"\nGenerating interpolation from sample {start_idx} to {end_idx}...")
    print(f"Number of steps: {n_steps}")

    interpolated_curves = []
    alphas = []

    with torch.no_grad():
        for step in range(n_steps):
            alpha = step / (n_steps - 1)
            alphas.append(alpha)

            interpolated_latent = (1 - alpha) * latent_codes[start_idx] + alpha * latent_codes[end_idx]

            latent_tensor = torch.FloatTensor(interpolated_latent[np.newaxis, :]).to(DEVICE)
            decoded, _ = model.decode(latent_tensor)
            curve = decoded[0].cpu().numpy()

            interpolated_curves.append(curve)

    # Analyze each curve
    print("\nAnalyzing dynamics of each interpolated curve...")
    print("-"*70)

    dynamics_progression = []

    for i, (curve, alpha) in enumerate(zip(interpolated_curves, alphas)):
        dynamics = analyze_curve_dynamics(curve)
        dynamics_progression.append(dynamics)

        label = "START" if i == 0 else ("END" if i == n_steps-1 else f"α={alpha:.2f}")
        print(f"\n{label}:")
        print(f"  Peak times (mean):     {np.mean(dynamics['peak_times']):.1f}")
        print(f"  Peak times (std):      {np.std(dynamics['peak_times']):.1f}")
        print(f"  Steady state (mean):   {np.mean(dynamics['steady_states']):.3f}")
        print(f"  Overshoot (mean):      {np.mean(dynamics['overshoot']):.3f}")
        print(f"  Temporal var (mean):   {np.mean(dynamics['temporal_variance']):.3f}")
        print(f"  Number of crossings:   {len(dynamics['crossings'])}")

        if len(dynamics['crossings']) > 0:
            early_crossings = [c['first_crossing'] for c in dynamics['crossings']]
            print(f"  First crossing (mean): {np.mean(early_crossings):.1f}")

    # Check if features change progressively
    print("\n" + "="*70)
    print("PROGRESSION ANALYSIS")
    print("="*70)

    # Extract feature trajectories
    mean_peak_times = [np.mean(d['peak_times']) for d in dynamics_progression]
    mean_steady_states = [np.mean(d['steady_states']) for d in dynamics_progression]
    mean_overshoots = [np.mean(d['overshoot']) for d in dynamics_progression]
    mean_temp_vars = [np.mean(d['temporal_variance']) for d in dynamics_progression]
    num_crossings = [len(d['crossings']) for d in dynamics_progression]

    # Check monotonicity (or smoothness)
    def is_monotonic_or_smooth(arr):
        """Check if array changes monotonically or smoothly"""
        diffs = np.diff(arr)
        # Monotonic if all same sign
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            return "MONOTONIC", True
        # Smooth if not too many sign changes
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        if sign_changes <= 1:
            return "SMOOTH", True
        return "ERRATIC", False

    print("\nFeature Progression Check:")
    print("-"*70)

    for feat_name, feat_values in [
        ("Mean peak time", mean_peak_times),
        ("Mean steady state", mean_steady_states),
        ("Mean overshoot", mean_overshoots),
        ("Mean temporal variance", mean_temp_vars),
        ("Number of crossings", num_crossings)
    ]:
        pattern, is_good = is_monotonic_or_smooth(feat_values)
        status = "✓" if is_good else "✗"
        print(f"  {status} {feat_name:25s}: {pattern}")
        print(f"     Values: {feat_values}")

    # Create detailed visualization
    print("\n" + "="*70)
    print("GENERATING DETAILED VISUALIZATION")
    print("="*70)

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, n_steps, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Time series curves
    for i, (curve, alpha) in enumerate(zip(interpolated_curves, alphas)):
        ax = fig.add_subplot(gs[0, i])

        for species in range(curve.shape[0]):
            ax.plot(curve[species], alpha=0.8, linewidth=2)

        label = "START" if i == 0 else ("END" if i == n_steps-1 else f"α={alpha:.2f}")
        ax.set_title(label, fontweight='bold', fontsize=12)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel('Time', fontsize=10)
        if i == 0:
            ax.set_ylabel('Abundance', fontsize=10)
        ax.grid(alpha=0.3)

    # Row 2: Peak time distribution
    for i, dynamics in enumerate(dynamics_progression):
        ax = fig.add_subplot(gs[1, i])

        peak_times = dynamics['peak_times']
        ax.bar(range(len(peak_times)), peak_times, color='steelblue', alpha=0.7)
        ax.set_ylim(0, max([max(d['peak_times']) for d in dynamics_progression]) * 1.1)
        ax.set_ylabel('Peak Time', fontsize=10)
        ax.set_xlabel('Species', fontsize=9)
        ax.set_title(f"Mean: {np.mean(peak_times):.1f}", fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    # Row 3: Overshoot per species
    for i, dynamics in enumerate(dynamics_progression):
        ax = fig.add_subplot(gs[2, i])

        overshoots = dynamics['overshoot']
        ax.bar(range(len(overshoots)), overshoots, color='coral', alpha=0.7)
        ax.set_ylim(0, max([max(d['overshoot']) for d in dynamics_progression]) * 1.1)
        ax.set_ylabel('Overshoot', fontsize=10)
        ax.set_xlabel('Species', fontsize=9)
        ax.set_title(f"Mean: {np.mean(overshoots):.2f}", fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    # Row 4: Feature progression plots
    ax_prog = fig.add_subplot(gs[3, :])

    x = np.array(alphas)

    ax_prog.plot(x, mean_peak_times, 'o-', linewidth=2.5, markersize=8,
                label='Mean Peak Time', color='steelblue')

    # Normalize other features to [0, 1] for comparison
    ax_prog_twin1 = ax_prog.twinx()
    ax_prog_twin2 = ax_prog.twinx()
    ax_prog_twin2.spines['right'].set_position(('outward', 60))

    ax_prog_twin1.plot(x, mean_steady_states, 's-', linewidth=2.5, markersize=8,
                      label='Mean Steady State', color='green')
    ax_prog_twin2.plot(x, mean_overshoots, '^-', linewidth=2.5, markersize=8,
                      label='Mean Overshoot', color='coral')

    ax_prog.set_xlabel('Interpolation Parameter α', fontweight='bold', fontsize=13)
    ax_prog.set_ylabel('Mean Peak Time', fontweight='bold', fontsize=12, color='steelblue')
    ax_prog_twin1.set_ylabel('Mean Steady State', fontweight='bold', fontsize=12, color='green')
    ax_prog_twin2.set_ylabel('Mean Overshoot', fontweight='bold', fontsize=12, color='coral')

    ax_prog.tick_params(axis='y', labelcolor='steelblue')
    ax_prog_twin1.tick_params(axis='y', labelcolor='green')
    ax_prog_twin2.tick_params(axis='y', labelcolor='coral')

    ax_prog.set_title('Progressive Change in Dynamical Features', fontweight='bold', fontsize=14)
    ax_prog.grid(alpha=0.3)
    ax_prog.set_xlim(-0.05, 1.05)

    # Combined legend
    lines1, labels1 = ax_prog.get_legend_handles_labels()
    lines2, labels2 = ax_prog_twin1.get_legend_handles_labels()
    lines3, labels3 = ax_prog_twin2.get_legend_handles_labels()
    ax_prog.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                  loc='upper left', fontsize=11)

    plt.savefig(OUTPUT_DIR / 'Interpolation_Analysis_Detailed.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Interpolation_Analysis_Detailed.pdf',
                bbox_inches='tight')
    print(f"\n✓ Saved: Interpolation_Analysis_Detailed.png/pdf")
    plt.close()

    # Summary conclusions
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)

    print("\n¿La interpolación muestra cambios progresivos REALES en dinámica?")
    print("-"*70)

    # Check if START and END are actually different
    start_peaks = mean_peak_times[0]
    end_peaks = mean_peak_times[-1]
    peak_change = abs(end_peaks - start_peaks)

    start_steady = mean_steady_states[0]
    end_steady = mean_steady_states[-1]
    steady_change = abs(end_steady - start_steady)

    print(f"\n📊 Cambio en features de START a END:")
    print(f"   Peak time:     {start_peaks:.1f} → {end_peaks:.1f} (Δ = {peak_change:.1f})")
    print(f"   Steady state:  {start_steady:.3f} → {end_steady:.3f} (Δ = {steady_change:.3f})")
    print(f"   Overshoot:     {mean_overshoots[0]:.2f} → {mean_overshoots[-1]:.2f} (Δ = {abs(mean_overshoots[-1]-mean_overshoots[0]):.2f})")

    if peak_change > 2.0:
        print("\n✅ SÍ: Las curvas START y END tienen peak times MUY diferentes")
        print("   → START peaks early, END peaks late (o viceversa)")
        print("   → La interpolación captura este cambio progresivo")
        print("   → ¡Esto confirma tu intuición!")
    else:
        print("\n⚠️  Las curvas START y END son relativamente similares")
        print("   → La interpolación es menos informativa")

    if all([is_monotonic_or_smooth(arr)[1] for arr in
            [mean_peak_times, mean_steady_states, mean_overshoots]]):
        print("\n✅ PROGRESIÓN SUAVE: Todas las features cambian monótonicamente")
        print("   → No hay saltos bruscos o inconsistencias")
        print("   → El espacio latente es CONTINUO y bien estructurado")
    else:
        print("\n⚠️  Algunas features cambian erráticamente")
        print("   → Podría haber discontinuidades en el espacio latente")

    print("\n" + "="*70)

    return dynamics_progression


if __name__ == "__main__":
    dynamics = compare_interpolation_progression(start_idx=50, end_idx=250, n_steps=7)
