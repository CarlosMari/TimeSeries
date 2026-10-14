#!/bin/bash
# Master script to regenerate ALL figures with 30D latent model
# All figures will have "_30" suffix to avoid overwriting 50D results

echo "================================================================================"
echo "REGENERATING ALL FIGURES WITH 30D LATENT MODEL"
echo "Model: model_ckpts/model_final_30_conditioned.pth"
echo "Latent dimension: 30 (down from 50)"
echo "Output suffix: _30"
echo "================================================================================"
echo ""

# Activate virtual environment
source TimeSeries/bin/activate

# Set environment variable for scripts to use
export LATENT_DIM=30
export MODEL_SUFFIX="_30"

echo "[1/8] Reconstruction Quality Analysis..."
python -c "
import sys
sys.path.insert(0, '.')

# Configuration for 30D model
model_config = {
    'in_channels': 7,
    'n_curves': 7,
    'seq_len': 65,
    'latent_dim': 30,
    'rnn_hidden_size': 256,
    'rnn_num_layers': 2,
    'scale_prediction_mode': 'log',
    'use_scale_conditioning': True,
}

MODEL_PATH = 'model_ckpts/model_final_30_conditioned.pth'
OUTPUT_SUFFIX = '_30'

exec(open('quick_recon_30d.py').read())
" 2>&1 | grep -v "^$"

echo ""
echo "[2/8] Variance Explained (PCA)..."
python quick_variance_30d.py

echo ""
echo "[3/8] Latent Collapse Analysis..."
python quick_collapse_30d.py

echo ""
echo "[4/8] Controlled Generation Examples..."
python quick_generation_30d.py

echo ""
echo "[5/8] Latent Space Structure (t-SNE/UMAP)..."
python quick_latent_30d.py

echo ""
echo "[6/8] Phase Space Comparisons..."
python quick_phase_30d.py

echo ""
echo "[7/8] Controlled Dynamics Analysis..."
python quick_controlled_30d.py

echo ""
echo "[8/8] Recurrence Dynamics..."
python quick_recurrence_30d.py

echo ""
echo "================================================================================"
echo "ALL 30D FIGURES GENERATED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "Check 'final figures/' directory for all files with '_30' suffix"
echo ""
ls -lh "final figures/"/*_30.png | wc -l | xargs echo "Total PNG files generated:"
echo ""
echo "You can now compare 30D vs 50D models side-by-side!"
echo "================================================================================"
