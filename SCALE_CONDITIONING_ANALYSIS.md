# Scale Conditioning: Adding Max Values as Input

**Date:** 2025-11-28
**Proposal:** Add max_vals as input, project to latent space (50D), keep architecture mostly the same

---

## 🎯 The Idea

**Current:**
```
X (normalized) → Encoder → z (50D) → Decoder → X_hat
                            ↓
                      Max Predictor → max_vals_pred (often wrong)
```

**Proposed:**
```
X (normalized) ────→ Shape Encoder ─→ z_shape (50D) ──┐
                                                       ├─→ Combine → z (50D) → Decoder → X_hat
max_vals ──────────→ Scale Encoder ─→ z_scale (50D) ──┘                           ↓
                                                                          Max Predictor → max_vals_pred
```

---

## ✅ Why This is Clever

### 1. **Solves Information Bottleneck**
- Max values contain EXACT scale information
- By encoding them into latent space, we guarantee this info is available
- Model learns to use this information for reconstruction

### 2. **Avoids Step 2 Problem**
You mentioned Step 2 struggles with low-scale curves. This is because:
- Step 2: Curves have peaks ranging from 0.1 to 1.0
- LSTM must handle 10x dynamic range
- Small curves get "washed out" (gradient ~0.1x smaller)
- Harder to learn patterns at different scales simultaneously

**With conditioning:**
- Input stays normalized (Step 3) → All curves equally learnable ✓
- Scale info provided separately → No dynamic range issues ✓

### 3. **Interpretable Latent Space**
- `z_shape` captures temporal dynamics
- `z_scale` captures amplitude information
- Combined `z` has both → Better disentanglement

---

## ⚠️ Key Challenge: The Chicken-and-Egg Problem

**At training time:**
```python
# We have true max_vals ✓
max_vals_true → z_scale → z → X_hat (works great!)
```

**At generation time:**
```python
# We DON'T have max_vals yet ❌
??? → z_scale → z → X_hat (what goes here?)
```

**The circular dependency:**
1. We need `max_vals` to create `z`
2. We predict `max_vals` from `z`
3. But we need `z` to predict `max_vals`!

---

## 🔧 Solution Architectures

### **Option 1: Two-Stage Generation (RECOMMENDED)**

**Stage 1:** Predict max_vals from shape encoding only
**Stage 2:** Use predicted max_vals for final generation

```python
class CVAE_TwoStage(nn.Module):
    def __init__(self, config):
        # Shape encoder (from X)
        self.shape_encoder = BiLSTM(input=7, hidden=256, layers=2)

        # Scale encoder (from max_vals)
        self.scale_encoder = nn.Sequential(
            nn.Linear(7, 50),
            nn.SiLU(),
            nn.Linear(50, 50)
        )

        # VAE components (operate on combined encoding)
        self.fc_mu = nn.Linear(100, 50)  # Input: [z_shape, z_scale]
        self.fc_log_var = nn.Linear(100, 50)

        # Decoder (gets z + scale conditioning)
        self.decoder_hidden = nn.Linear(50, 512)  # z → LSTM hidden
        self.decoder_cell = nn.Linear(50, 512)
        self.decoder_lstm = LSTM(input=7+50, hidden=256)  # [curves, z_scale]

        # Scale predictor (ONLY from shape encoding!)
        self.scale_predictor = nn.Sequential(
            nn.Linear(50, 25),
            nn.SiLU(),
            nn.Linear(25, 6)  # Predict curves 1-6
        )

    def forward(self, X, max_vals_true):
        """Training: Use true max_vals."""
        # Encode shape
        z_shape = self.shape_encoder(X)  # (N, 50)

        # Encode scale
        z_scale = self.scale_encoder(max_vals_true)  # (N, 50)

        # Combine for VAE
        z_combined = torch.cat([z_shape, z_scale], dim=1)  # (N, 100)
        mu = self.fc_mu(z_combined)
        log_var = self.fc_log_var(z_combined)
        z = self.sample(mu, log_var)  # (N, 50)

        # Predict max_vals (from shape encoding ONLY)
        max_vals_pred = self.scale_predictor(z_shape)

        # Decode (condition on z_scale throughout)
        X_hat = self.decode(z, z_scale)

        return X_hat, mu, log_var, max_vals_pred

    def generate(self, num_samples, device):
        """Generation: Two-stage process."""
        # Stage 1: Sample shape and predict scales
        z_shape_prior = torch.randn(num_samples, 50, device=device)
        max_vals_pred = self.scale_predictor(z_shape_prior)

        # Stage 2: Encode predicted scales and generate
        z_scale = self.scale_encoder(max_vals_pred)
        z_combined = torch.cat([z_shape_prior, z_scale], dim=1)

        # Sample from VAE
        mu = self.fc_mu(z_combined)
        log_var = self.fc_log_var(z_combined)
        z = self.sample(mu, log_var)

        # Decode
        X_gen = self.decode(z, z_scale)

        return X_gen, max_vals_pred
```

**Pros:**
- ✅ Clear separation: Shape predictor doesn't see scale input
- ✅ Generation is well-defined (two stages)
- ✅ Scale predictor learns from shape features only
- ✅ Can still improve scale prediction independently

**Cons:**
- ⚠️ Predicted scales might not match what was used in training
- ⚠️ Error in stage 1 propagates to stage 2

---

### **Option 2: Iterative Refinement**

```python
def generate_iterative(self, num_samples, device, n_iters=3):
    """Iteratively refine max_vals and reconstruction."""
    # Initialize with prior mean
    max_vals = torch.ones(num_samples, 7, device=device) * 0.5

    for _ in range(n_iters):
        # Encode current max_vals estimate
        z_scale = self.scale_encoder(max_vals)

        # Sample shape
        z_shape = torch.randn(num_samples, 50, device=device)

        # Combine and generate
        z_combined = torch.cat([z_shape, z_scale], dim=1)
        mu = self.fc_mu(z_combined)
        z = self.sample(mu, torch.zeros_like(mu))  # Low variance

        # Decode
        X_gen = self.decode(z, z_scale)

        # Re-predict max_vals from generated X
        z_shape_new = self.shape_encoder(X_gen)
        max_vals = self.scale_predictor(z_shape_new)

    return X_gen, max_vals
```

**Pros:**
- ✅ Self-correcting (refines prediction)
- ✅ Converges to consistent max_vals

**Cons:**
- ⚠️ Slower (3x forward passes)
- ⚠️ Might not converge

---

### **Option 3: Adversarial/GAN-like**

Train a discriminator that checks if (X, max_vals) pair is realistic.

**Pros:**
- ✅ Forces consistency

**Cons:**
- ⚠️ Much harder to train
- ⚠️ Overkill for this problem

---

## 🎨 Implementation Details

### **How to Combine z_shape and z_scale?**

#### **Option A: Concatenate then Project** (RECOMMENDED)
```python
z_combined = torch.cat([z_shape, z_scale], dim=1)  # (N, 100)
mu = nn.Linear(100, 50)(z_combined)
log_var = nn.Linear(100, 50)(z_combined)
z = sample(mu, log_var)
```

**Pros:**
- Allows model to learn optimal mixing
- No information loss
- More expressive

**Cons:**
- Doubles the input dimension to bottleneck

#### **Option B: Element-wise Addition**
```python
z_combined = z_shape + z_scale  # (N, 50)
mu = nn.Linear(50, 50)(z_combined)
```

**Pros:**
- Keeps dimension (50)
- Simpler

**Cons:**
- Information loss (can't distinguish which features are shape vs scale)
- Less expressive

#### **Option C: Weighted Sum**
```python
alpha = nn.Parameter(torch.tensor(0.5))  # Learnable
z_combined = alpha * z_shape + (1 - alpha) * z_scale
```

**Pros:**
- Learns optimal balance

**Cons:**
- Too simplistic

**Recommendation:** Use Option A (concatenate then project)

---

### **Should the Decoder See z_scale?**

**Option 1: Decoder input includes z_scale** (RECOMMENDED)
```python
# At each timestep:
decoder_input = torch.cat([previous_output, z_scale], dim=-1)
# Input dim: 7 (curves) + 50 (z_scale) = 57
```

**Pros:**
- ✅ Decoder can directly use scale information
- ✅ Better reconstruction of amplitudes
- ✅ More control over output scale

**Cons:**
- ⚠️ Changes decoder architecture (minor)

**Option 2: Decoder only sees z (scale info is mixed in)**
```python
# Current architecture:
decoder_input = torch.cat([previous_output, z], dim=-1)
# Input dim: 7 + 50 = 57 (same as before!)
```

**Pros:**
- ✅ No architecture change needed
- ✅ Simpler

**Cons:**
- ⚠️ Scale info might be diluted in z

**Recommendation:** Try Option 1 first for better performance, fall back to Option 2 if needed

---

## 📊 Expected Performance

### **Reconstruction Quality:**
```
Current:  R² = 0.946 (normalized)
Expected: R² = 0.96+ (slightly better due to scale conditioning)
```

### **Max Value Prediction:**
```
Current:  R² = -0.28 (terrible)
Expected: R² = 0.40 - 0.60 (much better, but still limited by shape-scale correlation)
```

**Why not higher?**
- The scale predictor still only sees z_shape (by design)
- z_shape is derived from normalized X (no scale info)
- Shape-scale correlation is still only 0.13
- But: Model can learn better features through end-to-end training

### **Generation Quality:**
```
Current:  Generated curves often have wrong scales
Expected: Scales will be more realistic and consistent
```

---

## ⚖️ Trade-offs

### **Advantages:**
1. ✅ **Better reconstruction** - Model has access to true scales
2. ✅ **No dynamic range issues** - Keep normalized input
3. ✅ **Interpretable latent space** - Separate shape/scale encodings
4. ✅ **Conditional generation** - Can control scales at generation time
5. ✅ **End-to-end training** - Scale predictor learns better features

### **Disadvantages:**
1. ⚠️ **More complex** - Additional encoder and combination logic
2. ⚠️ **Train/test gap** - Must use predicted scales at test time
3. ⚠️ **Hyperparameter tuning** - Need to balance multiple objectives
4. ⚠️ **Slower generation** - Two-stage process

---

## 🚀 Recommended Implementation Plan

### **Phase 1: Minimal Change (Week 1)**

1. Add scale encoder:
```python
self.scale_encoder = nn.Sequential(
    nn.Linear(7, 50),
    nn.SiLU(),
    nn.Linear(50, 50)
)
```

2. Modify bottleneck:
```python
# In forward():
z_shape = self.encoder(X)
z_scale = self.scale_encoder(max_vals)
z_combined = torch.cat([z_shape, z_scale], dim=1)
mu = self.fc_mu(z_combined)  # Change from 50→50 to 100→50
```

3. Keep decoder unchanged (z already contains scale info)

4. Keep scale predictor on z_shape only

5. Train and evaluate

### **Phase 2: Decoder Conditioning (Week 2)**

If Phase 1 works but reconstruction isn't great:

1. Modify decoder to also see z_scale:
```python
decoder_input = torch.cat([previous_output, z, z_scale], dim=-1)
# Input: 7 + 50 + 50 = 107
```

2. Retrain

### **Phase 3: Advanced (Optional)**

- Try iterative refinement for generation
- Add adversarial loss for (X, max_vals) consistency
- Experiment with different combination strategies

---

## 🧪 Validation Tests

After implementation, verify:

```python
# 1. Training works
assert train_loss decreases over epochs

# 2. Reconstruction improves with scale conditioning
recon_loss_with_true_scales < recon_loss_without_scales

# 3. Max value prediction improves
r2_new > r2_old (hopefully > 0)

# 4. Generated samples are realistic
X_gen, max_vals_gen = model.generate(1000)
assert (X_gen >= 0).all() and (X_gen <= 1).all()
assert (max_vals_gen >= 0).all() and (max_vals_gen <= 1).all()

# 5. Two-stage generation is consistent
# Generate twice with same random seed, should get same result
torch.manual_seed(42)
X1, max1 = model.generate(10)
torch.manual_seed(42)
X2, max2 = model.generate(10)
assert torch.allclose(X1, X2)
```

---

## 🎯 My Verdict

**Is this a good idea?** **YES!** ✅

**Why:**
1. Solves the information bottleneck elegantly
2. Avoids Step 2's dynamic range problems
3. Creates a more principled architecture (CVAE)
4. Enables controlled generation
5. Should improve both reconstruction AND scale prediction

**Concerns:**
1. Train/test gap (but manageable with two-stage generation)
2. Added complexity (but worth it)
3. May not fully solve scale prediction (still limited by shape-scale correlation)

**Expected outcome:**
- Max value R² improvement: **-0.28 → 0.40-0.60**
- Reconstruction R² improvement: **0.946 → 0.96+**
- Generation quality: **Much better scale consistency**

**Better than Step 2 input?**
- For reconstruction: Probably similar
- For scale prediction: Potentially better (end-to-end learning)
- For training stability: **YES** (no dynamic range issues)

---

## 🔧 Quick Start Code

```python
# Modify LSTM_VAE.__init__():
self.scale_encoder = nn.Sequential(
    nn.Linear(7, self.latent_dim),
    nn.SiLU(),
    nn.Linear(self.latent_dim, self.latent_dim)
)

# Change bottleneck dimensions:
encoder_output_dim = self.rnn_num_layers * 2 * self.rnn_hidden_size
self.fc_mu = nn.Linear(encoder_output_dim + self.latent_dim, self.latent_dim)
self.fc_log_var = nn.Linear(encoder_output_dim + self.latent_dim, self.latent_dim)

# Modify forward():
def forward(self, X, max_vals, teacher_forcing_ratio=0.5):
    # Encode shape
    _, (h_n, _) = self.encoder_rnn(X.permute(0, 2, 1))
    z_shape = h_n.permute(1, 0, 2).reshape(batch_size, -1)

    # Encode scale
    z_scale = self.scale_encoder(max_vals)

    # Combine
    z_combined = torch.cat([z_shape, z_scale], dim=1)
    mu = self.fc_mu(z_combined)
    log_var = self.fc_log_var(z_combined)
    z = self.sample(mu, log_var)

    # Predict max_vals from shape encoding only
    max_vals_pred = self.max_value_predictor(z_shape)

    # Decode (rest is same)
    ...
```

---

**Status:** Ready to implement! This is a solid architectural improvement.
