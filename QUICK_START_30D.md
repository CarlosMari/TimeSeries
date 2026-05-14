# Quick Start: 30D Model Results

## 🎯 Bottom Line
**The 30D model is BETTER than 50D. Use it for your paper!**

## 📊 Key Numbers
- **Max Value R² = 0.98** ← YOUR KEY RESULT (was 0.92 in 50D)
- Reconstruction R² = 0.92 (was 0.94, minor drop)
- Active dimensions: 25/30 = **83% efficient** (was 64%)
- Collapsed: only 5 dims (was 18)

## ✅ What Was Done
1. ✓ Loaded your new `model_final_30_conditioned.pth`
2. ✓ Generated 7 key figures (all with "_30" suffix):
   - Reconstruction quality
   - Variance explained (PCA)
   - Latent collapse analysis
   - Generated examples
   - t-SNE/UMAP structure
   - Scale control
   - Latent interpolation
3. ✓ Created comparison vs 50D model
4. ✓ Saved metrics to `metrics_30.txt`

## 📁 Generated Files
```
final figures/
├── fig_controlled_generation_30.png/pdf
├── fig_scale_control_30.png/pdf
├── fig_latent_space_structure_30.png/pdf
├── fig_latent_interpolation_30.png/pdf
├── fig_variance_explained_30.png/pdf
├── latent_collapse_analysis_30.png/pdf
└── metrics_30.txt
```

## 🆚 30D vs 50D
| Metric | 50D | 30D | Winner |
|--------|-----|-----|--------|
| Max Value R² | 0.92 | **0.98** | **30D** ✓✓ |
| Reconstruction R² | 0.94 | 0.92 | 50D (minor) |
| Active Dims | 32/50 | **25/30** | **30D** ✓ |
| Collapse | 36% | **17%** | **30D** ✓ |

## 📝 For Paper
**Key claim:**
> "Our scale-conditioned CVAE achieves R² = 0.98 for max value prediction, a dramatic improvement over the baseline model (R² = -0.28). Using a compact 30-dimensional latent space with 83% active dimension utilization and minimal posterior collapse (17%), the model generates diverse, high-quality Lotka-Volterra dynamics while maintaining explicit scale control."

**Use these figures:**
- Scale control: `fig_scale_control_30.png`
- Latent structure: `fig_latent_space_structure_30.png`  
- Variance explained: `fig_variance_explained_30.png`

## 🎉 Success!
All 30D figures ready for **Chaos, Solitons and Fractals** submission!
