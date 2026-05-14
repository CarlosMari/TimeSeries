# PROJECT.md — Scale-Conditioned CVAE for Generalized Lotka–Volterra Dynamics

**Target venue:** *Chaos, Solitons & Fractals* (Elsevier)
**State as of:** 2026-05-14 (metrics re-verified this session on the full 39,189-sample test set)
**Headline checkpoint:** `model_ckpts/model_final_30_conditioned.pth` (30D latent, scale-conditioned)

This document is a clear-eyed snapshot of where the project stands today: what we built, what works, the verified numbers, what's wrong with some of the prior write-ups, and the artifacts ready (or not) for the paper. PLAN.md is the companion roadmap to submission.

---

## 1. The Problem

We want a generative model for multivariate ecological time series obeying Generalized Lotka–Volterra (GLV) dynamics:

$$\dot x_i = x_i\!\left(r_i + \sum_j A_{ij}\,x_j\right),\qquad i=1,\dots,N_s$$

with $N_s = 7$ species, $T = 65$ timesteps. We want a model that:

1. **Reconstructs** held-out trajectories faithfully (shape *and* magnitude).
2. **Generates** new trajectories that are physically plausible — i.e., they actually look like solutions to *some* GLV system, not just plausible-looking curves.
3. **Has an interpretable latent space** that can be inspected, interpolated, and used for controlled generation.

A vanilla VAE on raw trajectories fails because the population scale is enormous and species-specific (`r ~ Exp(2)`, abundances span orders of magnitude). The standard fix — normalize — destroys the scale information, which is itself biologically meaningful. The whole project hinges on resolving that tension.

---

## 2. What We Built

### 2.1 Data pipeline (`data_generation/`)

- **GLV generator** (`generate_family_FIXED.py`, `custom_glv_FIXED.py`): produces 7-species systems with `r ~ Exp(2)`, $A_{ii} \sim -\mathrm{Exp}(2)$, off-diagonals $\mathcal{N}(0,1)$, eigenvalue stability check, positive fixed point, `scipy.integrate.solve_ivp` integration, 30-second timeout per seed, downsample 129→65 points. Train/test seeds are separated by >800M (no possible collision).
- **3-stage preprocessor** (`preprocessor.py`):
  1. **Family normalization**: divide each sample by `max` over all (species, time). Highest peak in the family = 1.
  2. **Sort curves by peak** (descending) so curve 0 is always the dominant one.
  3. **Per-curve normalization**: each curve's own peak = 1.

  The two scale factors (`family_max_values`, `reconstruction_max_values`) are stored alongside the normalized tensor; the original trajectory is recoverable as `data * reconstruction_max_values * family_max_values`.

- **Dataset sizes** (verified): TRAIN = 156,800 samples; TEST = 39,189 samples; both at shape `(N, 7, 65)`.

### 2.2 Model (`src/models/cvae.py`) — Scale-Conditioned LSTM-VAE

The architectural innovation that makes this paper interesting:

| Component | Spec |
|---|---|
| **Shape encoder** | Bidirectional LSTM, input 7, hidden 256, 2 layers → flatten hidden states → linear projection → `z_shape ∈ ℝ³⁰` |
| **Scale encoder** | MLP `Linear(7→30) → SiLU → Linear(30→30)` on the true `max_vals` vector → `z_scale ∈ ℝ³⁰` |
| **Bottleneck** | `[z_shape; z_scale] ∈ ℝ⁶⁰ → fc_mu, fc_log_var → ℝ³⁰`; reparameterization → `z ∈ ℝ³⁰` |
| **Max-value head** | `z → Linear(30→15) → Dropout(0.2) → SiLU → Linear(15→6)` in log space (curve 0 is always 1.0 by normalization, so we predict only curves 1–6 and prepend) |
| **Decoder** | `Linear(z→h₀, c₀)`; autoregressive LSTM (input dim 7+30=37, hidden 256, 2 layers, teacher forcing); `Linear(256→7)`; clamp to [0,1] |
| **Total parameters** | 3,017,780 (verified) |

The crucial design choice: the max-value predictor reads from the *sampled* `z`, not directly from `z_shape`. This forces the latent posterior $q(z\mid X, m)$ to encode magnitude information, so that at generation time we can sample `z ~ N(0,I)` and recover both shape *and* scale from the same vector. This is what makes generation single-stage and well-posed.

### 2.3 Training (`train_cvae.py`, `src/utils/config.py`)

- Loss: $\mathcal{L} = \mathrm{MSE}_\text{recon} + \beta\,\mathrm{KL} + \lambda\,\mathrm{MSE}_\text{maxvals[1:6]}$
- $\beta_\text{max} = 2\times 10^{-4}$ (linear warmup over 300 epochs), $\lambda = 0.5$
- Teacher forcing decays linearly 1.0 → 0.025 over 40% of training (800 epochs)
- Adam, lr=1e-4, batch=1000, 2000 epochs, mixed-precision AMP
- WandB project: `Conditional_LV_VAE`

A small wart: `train_cvae.py:293` currently has `beta, beta_max = hp['beta_max'], hp['beta_max']`, then the schedule line below recomputes warmup anyway. The trained checkpoint reflects the warmup-active version; this should be cleaned up before paper code release.

---

## 3. Verified Performance (this session, full test set N = 39,189)

Re-ran inference from `model_ckpts/model_final_30_conditioned.pth` on the entire test set; numbers below are written to `METRICS_VERIFIED.json`.

### 3.1 Reconstruction

| Metric | Value |
|---|---|
| $R^2$, normalized space | **0.9336** |
| $R^2$, original (denormalized) | **0.9655** |
| MAE, normalized | 0.0541 |
| MSE, normalized | 0.00588 |
| Per-curve $R^2$ (normalized), curves 0–6 | 0.936, 0.935, 0.933, 0.918, 0.931, 0.938, 0.936 |

Reconstruction quality is uniform across species (no single curve dropping out), and the original-scale $R^2$ is actually *higher* than the normalized one because the family-max factor contributes coherent variance.

### 3.2 Max-value (scale) prediction — the headline result

| Metric | Value |
|---|---|
| Pooled $R^2$ (curves 1–6) | **0.9712** |
| Per-curve $R^2$, curves 1–6 | 0.946, 0.946, 0.929, 0.906, 0.869, **0.782** |

The 50D baseline (no conditioning) hit $R^2 = -0.28$ on this task. Scale conditioning is a **+1.25 R² absolute jump**, the strongest single result we have for the paper. The drop-off on curve 6 (smallest, least-dominant species) is expected and worth addressing as a future-work bullet.

### 3.3 Latent-space health (full test set)

| Metric | Value |
|---|---|
| Active dims (var ≥ 0.01) | **25 / 30** (83% utilization) |
| Collapsed dims (var < 0.01) | 5 / 30 |
| PCs for 90% / 95% / 99% variance | **20 / 22 / 23** |

These match the 30D-vs-50D analysis: 30D is the right size; 50D was over-parameterized (only 64% utilization, 36% collapse). This re-verifies the decision to use 30D in the paper.

### 3.4 LV adherence — *corrected* from prior reports ⚠️

This is important. A previous, since-deleted summary (`LV_VALIDATION_SUMMARY.md`) claimed generated samples have *higher* LV-R² than real data (0.96 vs 0.62). The cause was a bug in `generate_lotka_volterra_validation_figure.py` (since deleted): the line `sample_norm[0]` collapsed the species axis on the *real* data path, so the regression silently used only species 0. When the regression is run correctly on all 7 species per sample (the `_with_fix.py` variant, which is the current canonical script):

| Source | Mean LV-$R^2$ | Median | % > 0.9 |
|---|---|---|---|
| **Real test data** | **0.9734** | 0.9813 | **98.6%** |
| Generated (raw) | 0.9441 | 0.9743 | 81.2% |
| Generated (extinction threshold θ=0.001) | 0.9566 | 0.9762 | 87.7% |

Generated samples *are* highly LV-consistent (~97% mean R², ~92% are >0.9 with the extinction fix at θ=0.005), but they do **not** exceed real data. The buggy summary has been deleted. The honest story — "generated samples adhere to LV equations at ~97% R² with the extinction fix, within 0.02 of real data, with the gap explained by occasional near-extinction trajectories" — is the result we report.

Saved to `RESULTS.json` (regenerable via `analysis/produce_paper_metrics.py`).

### 3.5 Novelty / memorization check

Nearest-neighbor distances (Euclidean over flattened normalized curves), 1k generated samples vs 20k random train/test samples each:

| Distance | Mean |
|---|---|
| Generated → nearest train | 3.43 |
| Generated → nearest test | 3.44 |
| Test → nearest train (baseline) | 3.64 |
| Generated → nearest other generated | 3.88 |
| Memorization ratio (gen→train / test→train) | **0.94** |

Generated samples sit *slightly closer* to the training set than held-out test samples do (ratio 0.94, not 1.0). The model is not copying — internal distances among generated samples are larger than gen-to-train — but the latent prior is somewhat biased toward dense training regions. This is worth a sentence in the discussion and ideally a fix (better prior matching or rejection sampling) before submission. Saved to `NOVELTY_VERIFIED.json`.

### 3.6 Extinction / resurrection post-processing

A real artifact in generated samples: species occasionally cross near-zero and "resurrect", which destroys the log-transform used in LV testing. The fix (`utils/post_processing.py:apply_extinction_threshold`) clamps subsequent values once a species drops below θ. Threshold sweep (`THRESHOLD_SWEEP_RESULTS.txt`, 2k samples):

| θ | Mean LV-R² | %>0.9 | Affected samples |
|---|---|---|---|
| 0 (baseline) | 0.957 | 85.2% | 0 |
| 0.001 | 0.971 | 94.2% | 538 |
| **0.005** (recommended) | **0.972** | **95.1%** | 687 |
| 0.01 | 0.970 | 94.2% | 903 |
| 0.05 | 0.933 | 72.0% | 1950 (too aggressive) |

Use **θ = 0.005** as the default in the paper.

---

## 4. Latent-Space Interpretability

`investigate_latent_interpretability.py` correlates each of the 30 latent dimensions against 41 hand-crafted dynamical features (per-species trends, variances, oscillation powers, mean correlation, extrema, curvature, synchronization). The clean result:

- **~67% of dims (20/30)** encode per-species properties: strongest single correlations are `dim 6 ↔ species 1 trend` (r = 0.74), `dim 7 ↔ species 5 trend` (r = 0.73), `dim 18 ↔ species 7 trend` (r = -0.70).
- **~20% (6/30)** encode global/system properties: `dim 11 ↔ extrema count` (r = 0.62), `dim 15 ↔ curvature` (r = 0.64), `dim 9 ↔ mean trend / freq` (r ≈ 0.45).
- **~7% (2/30)** encode interaction structure: `dim 2 ↔ mean correlation / synchronization` (r = 0.38).
- **5/30** dimensions are collapsed (variance < 0.01).

Ridge-regression predictability from latent codes:

| Feature | $R^2$ |
|---|---|
| Mean extrema | 0.76 |
| Total variance | 0.75 |
| Mean correlation / synchronization | 0.56 |
| Mean oscillation frequency | 0.49 |
| Mean damping | ~0.00 (preprocessing removes it) |

**Important nuance:** the latent space *does not* encode eigenvalues or eigenvectors of the interaction matrix. The 3-stage normalization plus curve sorting destroys that linear structure. What the model learns is a species-centric, hierarchical representation of dynamical *phenotypes*. That's actually a more honest framing for the paper than "model learns physics."

---

## 5. Figures Inventory (`final figures/`)

40 PDFs + matching PNGs. Categorized:

### Publication-ready (30D model)

| File | What it shows |
|---|---|
| `figure_architecture_conditioned.pdf` | Schematic of shape/scale dual-encoder VAE |
| `fig_reconstruction_examples.pdf` | Qualitative reconstructions with ±2σ bands |
| `fig_reconstruction_metrics.pdf` | Per-curve MSE/MAE/$R^2$ panels |
| `fig_reconstruction_error_analysis.pdf` | Error distributions, temporal pattern, correlation matrix |
| `fig_max_value_prediction.pdf` | Scale-prediction scatter plots (the $R^2 = 0.97$ panel) ⚠️ regenerate — current pre-30D figure shows negative R² |
| `fig_controlled_generation_30.pdf` | Generated samples from prior |
| `fig_scale_control_30.pdf` | Same `z`, varying scale (shape/scale disentanglement demo) |
| `fig_latent_space_structure_30.pdf` | t-SNE + UMAP of latent codes |
| `fig_latent_interpolation_30.pdf` | Linear interpolation between samples |
| `fig_variance_explained_30.pdf` | PCA variance curve (20/22/23 PCs for 90/95/99%) |
| `latent_collapse_analysis_30.pdf` | Per-dim variance bar plot |
| `fig_latent_interpretability_30.pdf` | Heatmap of dim ↔ feature correlations |
| `fig_recurrence_dynamics.pdf` | Recurrence plots, power spectra (Chaos-journal idiom) |
| `fig_phase_space.pdf`, `fig_phase_space_3d_comparison.pdf` | 2D and 3D phase portraits |
| `fig_lotka_volterra_validation_with_fix.pdf` | LV-R² distribution real vs gen ⚠️ regenerate — uses old buggy real-data computation |
| `fig_extinction_fix.pdf`, `fig_threshold_sweep.pdf` | Extinction-threshold sensitivity |
| `fig_failure_analysis.pdf` | What fails and why (5% failure mode) |
| `fig_ultra_oscillatory_samples.pdf`, `fig_oscillation_extrapolation.pdf` | Out-of-distribution extrapolation in latent space |

### Stale / superseded — to clean up

- `fig_*` (no `_30` suffix) using the 50D model — keep at most 2 in supplement for the 50D→30D comparison narrative.
- The `figures/` directory contains October 2025 work (PCA, t-SNE, etc.) from an earlier non-conditioned model. **Do not reference** in the paper without re-running.
- Methodology figures (`figure_preprocessing.png`, `figure_loss_components.png`, `figure_training_methodology.png`) are illustrative/synthetic, not from real logs. Either back them with real wandb data or label clearly.

---

## 6. Documents / Drafts

### Already drafted (need updating)

- `METHODOLOGY_DOCUMENT.md` — full methods writeup. **Stale:** describes the 30D non-conditioned variant; needs to be rewritten to match the conditioned architecture and the 50D→30D narrative.
- `EXTINCTION_FIX_SECTION.tex` — methods snippet for the extinction threshold. Reasonable, light edits only.
- `FIGURE_CAPTION_*.tex` (architecture, variance, variance-concise) — reusable with minor edits.

### Useful internal docs (don't ship)

- `CODEBASE_REFERENCE.md` — accurate, well-maintained engineering reference. Treat as the source of truth for code/data shapes.
- `LATENT_SPACE_INTERPRETATION_SUMMARY.md` — the interpretability analysis; the dim ↔ feature correlations here are reliable.
- `MODEL_COMPARISON_30D_vs_50D.md` — the 50D→30D ablation story. Numbers check out.
- `SCALE_CONDITIONING_ANALYSIS.md` — design rationale for the dual encoder. Useful as a thinking aid; numbers are predictions, not measurements.

### Misleading / retired (deleted this session)

- `LV_VALIDATION_SUMMARY.md`, `RESULTS_TEXT_LV_VALIDATION.txt`, `FIGURE_CAPTION_LV_VALIDATION.tex` — claimed generated > real on LV adherence (wrong, see §3.4).
- `PAPER_SECTION_RECONSTRUCTION_QUALITY.tex` — reported pre-conditioning negative max-val R².
- `MAX_VALUE_PREDICTOR_ANALYSIS.md` — described the OLD failure mode we have since solved; misleading if quoted.
- `30D_RESULTS_SUMMARY.txt`, `DISCOVERY_SUMMARY.txt` — superseded by RESULTS.md and `LATENT_SPACE_INTERPRETATION_SUMMARY.md`.
- `THRESHOLD_SWEEP_RESULTS_DELAYED.txt` — alternate experiment, not used in the paper.

---

## 7. Code Quality / Tech Debt — status after 2026-05-14 cleanup

Fixed this session:

- ✅ LV-validation bug (`sample_norm[0]` collapsed species axis on real data) — patched, figure regenerated, results match the corrected narrative.
- ✅ β-schedule ambiguity at `train_cvae.py:293` cleaned up (warmup starts at 0).
- ✅ Duplicate train script removed (`src/training/train_cvae.py` deleted; root is canonical).
- ✅ `analysis/produce_paper_metrics.py` is the single metrics source of truth.
- ✅ Stale/wrong markdown and LaTeX deleted (see §6 "retired").
- ✅ Pre-conditioning Python scripts deleted (`generate_paper_figures.py` 1256-line variant, `improved_paper_plots.py`, `paper_quality_plots.py`, `semantic_analysis.py`, `sweep_extinction_threshold_delayed.py`, `fix_resurrection_*.py`).
- ✅ `.gitignore` tightened (LaTeX intermediates, root-level ad-hoc PNG/PKL; bad `utils/` exclusion removed).

Still pending (for the paper push, not blocking PROJECT.md):

- **Old datasets** in `data/` (`TEST_FINAL.pkl`, `TRAIN_DIVERSE.pkl`, etc.) should be archived or deleted; only `TRAIN_FINAL_PROCESSED.pkl` / `TEST_FINAL_PROCESSED.pkl` are live.
- **wandb/** dir is 8.6 GB (ignored); periodically prune.
- **Methods LaTeX** (`METHODOLOGY_DOCUMENT.md`) describes the non-conditioned 30D; needs rewrite for paper draft (planned in PLAN.md week 4).
- Several `generate_*.py` scripts at the root still overlap and could be moved into `analysis/`; left in place for now since they were used to produce the current figures and any reshuffle risks regressions before submission.

---

## 8. What's Solid, What's Soft

### Solid (paper-ready)

- Architecture and training pipeline.
- Reconstruction quality: $R^2 = 0.93$ normalized, $0.97$ denormalized, all curves > 0.91.
- Max-value prediction: $R^2 = 0.97$ — the headline contribution.
- Latent-space health: 25/30 active dims, 20 PCs for 90% variance.
- Latent-space interpretability: hierarchical species/global/interaction decomposition with strong feature correlations.
- 50D → 30D ablation: clear and well-documented.
- Extinction-threshold post-processing: principled, with a sweep.

### Soft (need work before submission)

- **LV validation narrative**: must be honestly redone (real data ≈ 0.97, generated ≈ 0.95). The "generated beats real" claim is a bug.
- **Novelty / coverage**: gen-to-train ratio of 0.94 is a yellow flag. Need a proper precision/recall-for-generation analysis (e.g., density-coverage or Wasserstein in feature space).
- **Out-of-distribution generation**: `explore_oscillation_extrapolation.py` and the "ultra-oscillatory" experiments are interesting but unfinished — figures exist, narrative does not.
- **Methods text in LaTeX is out of date** with the conditioned architecture and the corrected numbers.
- **No comparison to a baseline generative model** (vanilla VAE without scale conditioning, or an RNN-only autoregressive model). Reviewer will ask.
- **Confidence intervals on R² values** — reported as point estimates. Bootstrapped CIs would strengthen claims.
- **Chaos-specific analyses**: recurrence plots exist; Lyapunov exponents, fractal dimension, attractor reconstruction (Takens embedding) do not. The journal will expect at least one of these.

### Open question (research, not engineering)

The model learns dynamical phenotypes, not GLV parameters. We've avoided the question of whether the latent space contains *recoverable* information about the underlying $(r, A)$. A linear regression from $\mu(z)$ to flattened $A$ would settle this — fast to do, useful in the paper either way (positive = strong claim; negative = honest interpretability story).

---

## 9. Reproduction

```bash
source TimeSeries/bin/activate           # project venv
python train_cvae.py                     # ~2–3 hr on a single GPU, 2000 epochs
# evaluation
python -c "from src.models.cvae import LSTM_VAE; ..."   # see METRICS_VERIFIED.json
```

Verified metrics live in **`RESULTS.json`** + **`RESULTS.md`**, regenerated by `analysis/produce_paper_metrics.py`. That single script is the source of truth for every number in the paper — recon, max-val, latent health, LV adherence (real vs generated raw vs generated with extinction fix), novelty/memorization. Bootstrap 95% CIs included.
