# Paper Figures Quick Reference

**Target:** Chaos, Solitons and Fractals | **Generated:** 2025-11-28

---

## 🎯 Top Priority Figures (Must Include)

1. **`figure_architecture_conditioned.pdf`** - Main architecture diagram
2. **`fig_scale_control.png`** - KEY INNOVATION: Scale control demo (R² jump -0.28 → 0.92)
3. **`fig_recurrence_dynamics.png`** - Perfect for Chaos journal (recurrence plots)
4. **`fig_real_vs_generated.png`** - Qualitative validation
5. **`fig_controlled_generation.png`** - Generation quality showcase

---

## 📊 All Figures by Category

### Architecture & Innovation
- `figure_architecture_conditioned.pdf` - Full architecture with LaTeX caption
- `fig_architecture_comparison.png` - Baseline vs Scale-conditioned comparison

### Generation Quality
- `fig_controlled_generation.png` - 6 diverse generated samples
- `fig_scale_control.png` - **STAR FIGURE** - Same shape, different scales
- `fig_real_vs_generated.png` - Side-by-side comparison

### Validation & Metrics
- `fig_reconstruction_examples.png` - Visual reconstruction quality
- `fig_reconstruction_metrics.png` - R² scores and distributions
- `fig_reconstruction_error_analysis.png` - Error patterns
- `fig_max_value_prediction.png` - R² = 0.92 (KEY RESULT!)

### Latent Space Analysis
- `fig_variance_explained.png` - PCA showing 25 PCs = 99% variance
- `fig_latent_space_structure.png` - t-SNE & UMAP projections
- `fig_latent_interpolation.png` - Smooth interpolations

### Dynamical Analysis (for Chaos journal)
- `fig_recurrence_dynamics.png` - Recurrence plots + power spectra
- `fig_phase_space.png` - 2D phase portraits

### Training & Optimization
- `fig_training_dynamics.png` - Loss curves, beta warmup, R² evolution
- `fig_architecture_comparison.png` - Performance comparison table

---

## 🔑 Key Results to Highlight

| Metric | Baseline | Scale-Conditioned | Improvement |
|--------|----------|-------------------|-------------|
| Max Value R² | **-0.28** | **0.92** | +1.20 |
| Reconstruction R² | 0.94 | 0.94 | Maintained |
| Active Latent Dims | ~40 | 32 | More efficient |
| Scale Control | ❌ No | ✅ Yes | NEW capability |

---

## 📝 One-Sentence Descriptions

1. **Architecture** → Dual encoding: shape + scale pathways
2. **Scale Control** → Same dynamics at different magnitudes (impossible w/ baseline)
3. **Recurrence** → Generated dynamics match real complexity
4. **Generation** → Diverse, high-quality Lotka-Volterra samples
5. **Validation** → R² = 0.92 for max values, 0.94 for reconstruction
6. **Latent Space** → 25 PCs explain 99%, well-structured, minimal collapse
7. **Phase Space** → Complex trajectories (spirals, oscillations)
8. **Interpolation** → Smooth, biologically plausible transitions

---

## 🚀 Regenerate All Figures

After retraining with `latent_dim=30`:

```bash
./generate_all_paper_figures.sh
```

Or individually:
```bash
python generate_reconstruction_quality_figures.py
python generate_variance_explained_figure.py
python generate_paper_figures_comprehensive.py
python generate_recurrence_dynamics_figure.py
python generate_training_analysis_figure.py
```

---

## 📂 File Organization

All figures saved in: **`final figures/`**

Format: Both `.png` (high-res) and `.pdf` (vector, preferred for publication)

Total: **14 distinct figures** (28 files)

---

## 💡 For Paper Writing

**Abstract:** Mention R² = 0.92 for max values (vs -0.28 baseline)

**Introduction:** Highlight scale control as main innovation

**Methods:** Use `figure_architecture_conditioned.pdf`

**Results:** Lead with `fig_scale_control.png`, then validation figures

**Discussion:** Emphasize recurrence dynamics, phase space complexity

**Supplementary:** Training curves, detailed metrics, interpolations

---

**See `PAPER_FIGURES_SUMMARY.md` for detailed captions and descriptions.**
