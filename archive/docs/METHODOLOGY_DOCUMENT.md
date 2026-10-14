# Generalized Lotka-Volterra Time Series Reconstruction using LSTM-VAE

## Abstract

We present an LSTM-based Variational Autoencoder (LSTM-VAE) architecture for learning and reconstructing Generalized Lotka-Volterra (GLV) ecological dynamics. The model encodes multi-species time series data into a continuous latent space and reconstructs normalized population trajectories while simultaneously predicting species-specific scaling factors through an auxiliary prediction head. This dual-objective approach enables the model to disentangle temporal dynamics from amplitude information, improving reconstruction quality and latent space interpretability.

---

## 1. Model Architecture

### 1.1 Overview

The LSTM-VAE consists of three primary components: (1) a bidirectional LSTM encoder that compresses input time series into a probabilistic latent representation, (2) a parallel multi-layer perceptron (MLP) that predicts scaling factors directly from the latent space, and (3) an autoregressive LSTM decoder that reconstructs the normalized time series (Figure 1).

**Architecture Diagram:**
![Model Architecture](figure_architecture.png)
*Figure 1: LSTM-VAE architecture showing encoder, latent bottleneck with parallel max value predictor, and autoregressive decoder.*

### 1.2 Encoder

The encoder processes input time series **X** ∈ ℝ^(N×7×65) where N is the batch size, 7 represents the number of species, and 65 denotes the temporal dimension. The input is permuted to shape (N, 65, 7) for sequential processing.

**Bidirectional LSTM Encoder:**
- Input size: 7 (number of species/curves)
- Hidden size: 256
- Number of layers: 2
- Bidirectional: Yes

The encoder produces hidden states from both forward and backward passes. The final hidden states from all layers and both directions are concatenated, yielding an encoded representation of dimension 2 × 2 × 256 = 1024.

### 1.3 Latent Space and Reparameterization

Two separate fully connected layers map the encoded representation to the latent distribution parameters:

- **μ** = FC(h_encoded) ∈ ℝ^30  (mean)
- **log σ²** = FC(h_encoded) ∈ ℝ^30  (log variance)

The latent vector **z** ∈ ℝ^30 is sampled using the reparameterization trick:

**z = μ + ε ⊙ σ**, where **ε ~ N(0, I)**

This formulation enables backpropagation through the stochastic sampling operation.

### 1.4 Auxiliary Max Value Predictor

A parallel MLP branch predicts the maximum abundance values for each species directly from the latent vector **z**. This auxiliary task encourages the latent space to encode amplitude information explicitly.

**Architecture:**
```
z (30) → FC(30 → 15) → Dropout(0.2) → SiLU → FC(15 → 6) → max_values (6)
```

**Scale Prediction Mode:** The predictor operates in logarithmic space to handle the wide range of abundance scales typical in ecological data. Predictions are made in log-space and then exponentiated:

- Forward: **ŷ_log** = MLP(z)
- Inverse: **ŷ** = exp(ŷ_log)

Note: Only 6 values are predicted (species 1-6). Species 0 is always normalized to 1.0 by construction (see Data Preprocessing).

### 1.5 Decoder

The decoder autoregressively generates the time series from the latent vector **z**.

**Initialization:** The decoder's initial hidden and cell states are derived from **z**:
- **h₀** = FC_h(z) ∈ ℝ^(L×256) reshaped to (2, N, 256)
- **c₀** = FC_c(z) ∈ ℝ^(L×256) reshaped to (2, N, 256)

where L = 2 is the number of LSTM layers.

**Autoregressive Decoding:**
At each time step t ∈ {1, ..., 65}, the decoder receives the concatenation of:
1. The previous output **x̂_{t-1}** ∈ ℝ^7 (or ground truth during teacher forcing)
2. The latent vector **z** ∈ ℝ^30

This conditioning ensures the decoder has access to the global latent context at every step.

**LSTM Decoder:**
- Input size: 7 + 30 = 37
- Hidden size: 256
- Number of layers: 2
- Bidirectional: No

**Output Projection:**
The decoder's hidden state at each step is mapped to the 7-dimensional output space and clamped to [0, 1]:

**x̂_t = clamp(FC(h_t), 0, 1)**

---

## 2. Data Preprocessing

### 2.1 Pipeline Overview

The preprocessing transforms raw GLV simulation data into a normalized format suitable for VAE training. This multi-stage normalization ensures consistent curve ordering and enables the model to learn shape information independently of scale.

**Preprocessing Pipeline:**
![Preprocessing Pipeline](figure_preprocessing.png)
*Figure 2: Multi-stage data preprocessing pipeline converting raw GLV trajectories to normalized inputs.*

### 2.2 Stage 1: Family-wise Normalization

**Input:** Raw GLV data **X_raw** ∈ ℝ^(N×7×65)

For each sample n, compute the global maximum across all species and time points:

**max_family[n] = max_{i,t} X_raw[n, i, t]**

Normalize each family by its global maximum:

**X_family[n] = X_raw[n] / max_family[n]**

**Purpose:** Ensures the highest peak in each sample equals 1.0, providing a consistent reference scale.

### 2.3 Stage 2: Curve Sorting by Peak Value

For each sample, compute the maximum value of each species curve:

**max_per_curve[n, i] = max_t X_family[n, i, t]**

Sort species indices in descending order of their maximum values:

**sorted_indices[n] = argsort(-max_per_curve[n])**

Reorder the curves:

**X_sorted[n] = X_family[n, sorted_indices[n]]**

**Purpose:** Establishes a canonical ordering where species are arranged by dominance, improving training consistency.

### 2.4 Stage 3: Individual Curve Normalization

Normalize each individual species curve to [0, 1]:

**max_values[n, i] = max_t X_sorted[n, i, t]**

**X_final[n, i] = X_sorted[n, i] / max_values[n, i]**

**Purpose:** Each curve's peak becomes exactly 1.0, allowing the model to focus on temporal shape rather than amplitude.

### 2.5 Data Package

The final preprocessed data package contains:
1. **data**: X_final ∈ ℝ^(N×7×65) - normalized time series for VAE input
2. **reconstruction_max_values** ∈ ℝ^(N×7) - per-curve scaling factors (target for auxiliary predictor)
3. **family_max_values** ∈ ℝ^N - global scaling factors for full reconstruction

---

## 3. Training Methodology

### 3.1 Loss Function

The training objective combines three components:

**ℒ_total = ℒ_recon + β · ℒ_KL + λ · ℒ_max_val**

#### 3.1.1 Reconstruction Loss
Mean squared error between reconstructed and original normalized curves:

**ℒ_recon = (1/N) Σ ||X̂ - X||²**

#### 3.1.2 KL Divergence
Regularization term enforcing a Gaussian prior on the latent space:

**ℒ_KL = -(1/2N) Σ_i Σ_d (1 + log σ²_d - μ²_d - σ²_d)**

where d indexes latent dimensions and i indexes samples.

#### 3.1.3 Max Value Prediction Loss
Mean squared error for scaling factor prediction, computed only for species 1-6 (species 0 is always 1.0):

**ℒ_max_val = (1/N) Σ ||max_pred[:, 1:6] - max_true[:, 1:6]||²**

Predictions are made in log-space and transformed back to original scale before computing the loss.

**Training Methodology:**
![Training Methodology](figure_training_methodology.png)
*Figure 3: Comprehensive training configuration including loss components, hyperparameters, and schedules.*

### 3.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 1×10⁻⁴ | Adam optimizer learning rate |
| Batch size | 1000 | Samples per training batch |
| Epochs | 2000 | Total training iterations |
| β_max | 2×10⁻⁴ | Maximum KL weight |
| λ | 0.5 | Max value loss weight |
| Latent dimension | 30 | Dimensionality of z |
| LSTM hidden size | 256 | Hidden units per LSTM layer |
| LSTM layers | 2 | Number of stacked LSTM layers |
| Weight decay | 0 | L2 regularization (disabled) |

### 3.3 Training Schedules

#### 3.3.1 β Warmup
The KL divergence weight β increases linearly from 0 to β_max over the first 300 epochs:

**β(epoch) = min(β_max, β_max · epoch / 300)**

**Rationale:** Prevents posterior collapse by initially prioritizing reconstruction quality.

#### 3.3.2 Teacher Forcing Decay
Teacher forcing probability decreases from 1.0 to 0.025 over 40% of training (800 epochs):

**TF_ratio(epoch) = max(0.025, 1.0 - 0.975 · epoch / 800)**

**Rationale:** Gradually transitions the decoder from supervised to autonomous generation.

### 3.4 Optimization

- **Optimizer:** Adam with default β₁=0.9, β₂=0.999
- **Mixed Precision:** Automatic mixed precision (AMP) training using CUDA
- **Gradient Scaling:** Dynamic loss scaling to prevent underflow in FP16

### 3.5 Evaluation Metrics

#### 3.5.1 Loss Components
- Reconstruction MSE
- KL divergence
- Max value prediction MSE

#### 3.5.2 Coverage Metric
For each test sample, the model generates K=10 reconstructions. The **coverage** measures the proportion of ground truth points falling within the ±2σ confidence interval:

**Coverage = (1/M) Σ 𝟙[X_true ∈ [μ - 2σ, μ + 2σ]]**

where M is the total number of points (batch_size × 7 species × 65 time steps), μ and σ are computed from the K samples, and 𝟙 is the indicator function.

**Expected coverage under calibrated uncertainty: ~95%**

**Loss Components:**
![Loss Components](figure_loss_components.png)
*Figure 4: Illustrative training dynamics showing the evolution of individual loss components during optimization.*

---

## 4. Implementation Details

### 4.1 Data Loading
- **Framework:** PyTorch DataLoader
- **Workers:** 4 parallel workers
- **Pin Memory:** Enabled for faster GPU transfer

### 4.2 Model Configuration
```python
model_config = {
    'n_curves': 7,
    'seq_len': 65,
    'latent_dim': 30,
    'rnn_hidden_size': 256,
    'rnn_num_layers': 2,
    'scale_prediction_mode': 'log'
}
```

### 4.3 Hardware
- **Device:** CUDA-enabled GPU
- **Precision:** Mixed FP16/FP32

---

## 5. Key Design Decisions

### 5.1 Parallel Max Value Prediction
By predicting scaling factors from the latent space rather than learning them implicitly, the model:
- Explicitly encodes amplitude information in z
- Enables controlled generation with specified scales
- Improves disentanglement between shape and magnitude

### 5.2 Logarithmic Scale Prediction
GLV dynamics exhibit exponential growth/decay, making log-space prediction more numerically stable and better aligned with the underlying process.

### 5.3 Fixed First Curve Maximum
Setting species 0's maximum to 1.0 by construction eliminates ambiguity in the normalization and reduces the prediction task dimensionality from 7 to 6, focusing learning on the relative scales of secondary species.

### 5.4 Autoregressive Decoder with Latent Injection
Concatenating z at every decoder step (rather than only initializing hidden states) provides stronger conditioning, improving long-sequence coherence.

### 5.5 Teacher Forcing Decay
Gradual reduction prevents train-test mismatch (exposure bias) while maintaining training stability in early epochs.

---

## 6. Data Generation Context

The input data consists of GLV dynamics simulated using the Runge-Kutta 4th-order method. The interaction matrices are generated using an elliptic normal distribution parameterized by correlation coefficient ρ and interaction strength α. The GLV system is defined as:

**dx_i/dt = x_i(r_i - x_i - Σ_j A_ij x_j)**

where:
- **x_i**: abundance of species i
- **r_i**: intrinsic growth rate of species i
- **A_ij**: interaction matrix (effect of species j on species i)

The model learns to reconstruct these dynamics from preprocessed trajectories.

---

## 7. Summary

This LSTM-VAE architecture combines sequential modeling with explicit scale prediction to learn interpretable latent representations of GLV dynamics. The multi-stage preprocessing ensures consistent input structure, while the auxiliary prediction head encourages meaningful latent organization. The training methodology employs principled scheduling of KL annealing and teacher forcing to balance reconstruction fidelity with latent space regularization. This framework provides a foundation for generative modeling, interpolation, and analysis of multi-species ecological time series.

---

## References

GLV functions adapted from:
- Clenet, M. (2022). *Elliptic model for ecological dynamics.* arXiv:2205.15591

## Appendix: File Structure

```
TimeSeries/
├── src/
│   ├── models/
│   │   └── cvae.py                    # LSTM-VAE implementation
│   └── utils/
│       └── config.py                   # Hyperparameter configuration
├── data_generation/
│   ├── glv_functions.py                # GLV dynamics and matrix generation
│   └── preprocessor.py                 # Data preprocessing pipeline
├── train_cvae.py                       # Main training script
└── data/
    ├── TRAIN_FINAL_PROCESSED.pkl       # Preprocessed training data
    └── TEST_FINAL_PROCESSED.pkl        # Preprocessed test data
```
