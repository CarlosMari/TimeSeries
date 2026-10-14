# Seed and Data Generation Issues - Analysis & Fixes

## Issues Found

### 1. **CRITICAL: First sample always uses seed=0**
**Location:** `data_generation/generate_family.py:26`

```python
sol = generate_curves_Mario(myseed=seed*i, ...)  # When i=0, this is always 0!
```

**Problem:** When `i=0`, `seed*i = 0` regardless of what `seed` is. This means:
- First sample in TRAIN: `74 * 0 = 0`
- First sample in TEST: `42 * 0 = 0`
- **Both datasets have identical first samples!**

**Impact:** Every dataset generated this way will have the same first sample.

---

### 2. **Seed Collisions between TRAIN and TEST**
**Found:** 28 collisions in first 1000 samples

Examples:
- TRAIN[0] uses seed 0, TEST[0] uses seed 0 ✗
- TRAIN[37] uses seed 2,738, TEST[66] uses seed 2,772 (close!)
- Pattern: Every 37 train samples approximately collide with test samples

**Math:**
- GCD(74, 42) = 2
- LCM(74, 42) = 1,554
- Collision pattern: `74*i = 42*j` when `j = 74i/42 = 37i/21`

This means train and test seeds overlap regularly, reducing dataset diversity.

---

### 3. **Random State Inconsistency**
**Locations:**
- `generate_family.py:17` - Sets global seed once
- `generate_family.py:26` - Calls `generate_curves_Mario` which resets seed internally
- `generate_family.py:29` - Uses global random state for noise

**Problem Flow:**
```python
np.random.seed(seed)  # Global seed set to 74

for i in range(num_curves):
    # This RESETS the global seed!
    sol = generate_curves_Mario(myseed=seed*i, ...)  # Internally: np.random.seed(seed*i)

    # This uses the RESET global state, not the original progression!
    noise = np.random.lognormal(...)
```

Each iteration resets the random state, so the noise isn't using a continuous random stream.

---

### 4. **Fallback Seed Logic Issue**
**Location:** `custom_glv.py:84-85`

```python
if c >= 500:
    np.random.seed(tc)
```

After 500 failed attempts to generate valid GLV parameters, it falls back to a global counter `tc`.
This can cause:
- Unpredictable seeds after many failures
- Potential seed reuse across different calls

---

## Recommended Fixes

### Fix 1: Use proper seed offsetting
```python
# Instead of:
sol = generate_curves_Mario(myseed=seed*i, ...)

# Use:
sol = generate_curves_Mario(myseed=seed + i, ...)
# Or better:
sol = generate_curves_Mario(myseed=seed*1000000 + i, ...)
```

### Fix 2: Use RandomState objects (Best Practice)
```python
def generate_data(num_curves, seed, name='TRAIN'):
    # Create independent random number generator
    rng = np.random.RandomState(seed)

    for i in pbar:
        # Pass the RNG, not just a seed
        sol = generate_curves_Mario(rng=rng, species=7, ...)

        # Use the SAME rng for noise
        noise = rng.lognormal(mean=0, sigma=SIGMA, size=shape)
```

### Fix 3: Ensure train/test separation
```python
TRAIN_SEED = 123456789  # Large, distinct seeds
TEST_SEED = 987654321
# OR use hash-based seeds:
TRAIN_SEED = hash("TRAIN") & 0xFFFFFFFF
TEST_SEED = hash("TEST") & 0xFFFFFFFF
```

### Fix 4: Remove global seed setting inside functions
Modify `generate_curves_Mario` to accept an `rng` parameter instead of setting global seed.

---

## Test Results

Current data files check (first 100 samples):
- ✓ No exact duplicates found
- ✓ No near-duplicates found
- ✓ Reasonable diversity metrics

**However**, seed collision issues remain and could cause:
- Reduced effective dataset size
- Train/test leakage
- Reproducibility issues

---

## Immediate Action Items

1. **Fix seed generation**: Change `seed*i` to `seed + i` or use RandomState
2. **Verify no duplicates**: Run full duplicate check on larger sample
3. **Update preprocessing**: Ensure consistency through pipeline
4. **Document**: Add seed usage to CLAUDE.md

---

## Code Example: Proper Implementation

```python
def generate_data_FIXED(num_curves, seed, name='TRAIN'):
    # Use RandomState for isolated RNG
    rng = np.random.RandomState(seed)

    sim_lists = []
    num_sols = 0
    pbar = tqdm(range(num_curves * 2), desc='Finding Correct Solutions')  # Overgenerate
    scaling_factor = float(np.exp(SIGMA**2 / 2))

    for i in pbar:
        if num_sols >= num_curves:
            break

        # Generate unique seed for this sample
        sample_seed = seed + 1000000 + i  # Large offset to avoid collisions

        sol = generate_curves_Mario(myseed=sample_seed, noise_level=0.01,
                                    species=7, tmax=20, n_points=129, plot_this=False)

        # Use independent RNG for noise
        shape = sol.shape
        noise = rng.lognormal(mean=0, sigma=SIGMA, size=shape)
        centered_noise = noise / scaling_factor
        sol = sol * centered_noise

        # Quality checks...
        if check_passes(sol):
            num_sols += 1
            sim_lists.append(sol)

    return np.array(sim_lists)
```
