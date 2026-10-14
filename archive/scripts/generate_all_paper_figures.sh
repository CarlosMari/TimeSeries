#!/bin/bash
# Master script to generate all paper figures for Chaos, Solitons and Fractals
# Run after model training to regenerate all publication-quality figures

echo "================================================================================"
echo "GENERATING ALL PAPER FIGURES"
echo "Scale-Conditioned CVAE for Lotka-Volterra Dynamics"
echo "Target Journal: Chaos, Solitons and Fractals"
echo "================================================================================"
echo ""

# Activate virtual environment
source TimeSeries/bin/activate

# Figure 1: Reconstruction Quality Analysis (4 sub-figures)
echo "→ Generating Figure Set 1: Reconstruction Quality Analysis..."
python generate_reconstruction_quality_figures.py
echo "✓ Complete"
echo ""

# Figure 2: Variance Explained (PCA Analysis)
echo "→ Generating Figure Set 2: Variance Explained..."
python generate_variance_explained_figure.py
echo "✓ Complete"
echo ""

# Figures 3-8: Comprehensive Paper Figures
echo "→ Generating Figure Set 3: Comprehensive Paper Figures (6 figures)..."
python generate_paper_figures_comprehensive.py
echo "✓ Complete"
echo ""

# Figure 9: Recurrence Dynamics
echo "→ Generating Figure Set 4: Recurrence & Dynamical Complexity..."
python generate_recurrence_dynamics_figure.py
echo "✓ Complete"
echo ""

# Figure 10-11: Training Analysis
echo "→ Generating Figure Set 5: Training Analysis & Comparison..."
python generate_training_analysis_figure.py
echo "✓ Complete"
echo ""

echo "================================================================================"
echo "ALL PAPER FIGURES GENERATED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "Output directory: final figures/"
echo ""
echo "Generated figures:"
echo "  • figure_architecture_conditioned.pdf - Architecture diagram"
echo "  • fig_controlled_generation.png/pdf - Generated examples"
echo "  • fig_scale_control.png/pdf - Scale control demonstration (KEY!)"
echo "  • fig_latent_space_structure.png/pdf - t-SNE & UMAP"
echo "  • fig_latent_interpolation.png/pdf - Latent interpolation"
echo "  • fig_phase_space.png/pdf - Phase space trajectories"
echo "  • fig_real_vs_generated.png/pdf - Real vs generated comparison"
echo "  • fig_recurrence_dynamics.png/pdf - Recurrence plots"
echo "  • fig_variance_explained.png/pdf - PCA variance"
echo "  • fig_training_dynamics.png/pdf - Training curves"
echo "  • fig_architecture_comparison.png/pdf - Innovation comparison"
echo "  • fig_reconstruction_examples.png/pdf - Reconstruction quality"
echo "  • fig_reconstruction_metrics.png/pdf - Metrics"
echo "  • fig_reconstruction_error_analysis.png/pdf - Error analysis"
echo "  • fig_max_value_prediction.png/pdf - Max value prediction"
echo ""
echo "Total: 14 figures (28 files with PNG + PDF formats)"
echo ""
echo "See PAPER_FIGURES_SUMMARY.md for detailed descriptions and suggested captions."
echo "================================================================================"
