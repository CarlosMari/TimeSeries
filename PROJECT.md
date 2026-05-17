# PROJECT.md — Comparative Evaluation of Generative Models of GLV Dynamics under a 3-Lens NLD Protocol

**Target venue:** *Chaos, Solitons & Fractals* (Elsevier), or comparable nonlinear-dynamics venue.
**State as of:** 2026-05-15 (paper pivoted; design doc at `docs/superpowers/specs/2026-05-15-comparative-evaluation-design.md`).
**Current headline (interim, single-model v1):** `model_ckpts/model_final_30_conditioned.pth` (30D, scale-conditioned). To be superseded by the 7-model comparative set once training completes (§A roadmap).

> **2026-05-15 PIVOT NOTICE.** Before this date the paper was framed around a single scale-conditioned VAE. A pre-Phase-4 audit concluded the contribution was incremental and the three "honest limitations" (§3.8, §3.9, §4.5) read as failures rather than findings. We have pivoted to **a comparative empirical study of 7 generative architectures on GLV trajectories, evaluated under a 3-lens nonlinear-dynamics protocol** (feature-MMD + RQA + Rosenstein λ₁). The model becomes one of seven worked examples; the *protocol* and the *comparison* are the contributions. The full rationale, locked decisions, and schedule are in the design doc above. **Existing numbers in §§3–4 below are still valid for the v1 single-model with the old preprocessing pipeline** (which included a sort-by-peak step). They will be re-derived on the new no-sort pipeline as Phase-A retraining completes; current numbers are preserved here as `v1 results` and will be archived to an appendix once the comparative table is finalized. **Until that table exists, treat everything below as v1 evidence informing the pivot, not as the final paper's results.**

---

## 1. The Problem

We want **a generative model for multivariate ecological time series** obeying Generalized Lotka–Volterra (GLV) dynamics:

$$\dot x_i = x_i\!\left(r_i + \sum_j A_{ij}\,x_j\right),\qquad i=1,\dots,N_s$$

with $N_s = 7$ species, $T = 65$ timesteps. We want generative models that:

1. **Reconstruct** held-out trajectories faithfully (shape *and* magnitude).
2. **Generate** new trajectories that are physically plausible — i.e., they actually look like solutions to *some* GLV system.
3. **Preserve nonlinear-dynamics invariants** of the underlying system (recurrence structure, Lyapunov exponent, dominant-frequency content). The point of the 2026-05-15 pivot is that **this third property is what we can no longer take for granted, and what our 3-lens evaluation protocol is built to measure**.

A vanilla VAE on raw trajectories fails because the population scale is enormous and species-specific (`r ~ Exp(2)`, abundances span orders of magnitude). The standard fix — normalize — destroys the scale information, which is itself biologically meaningful. **This single tension was the original paper's whole subject; in the pivoted paper it is the motivation for one of the seven comparators (the scale-conditioned VAE) and is no longer the central narrative.**

### 1.1 Why the pivot, in one paragraph

We trained a state-of-the-art scale-conditioned VAE; it reconstructs (R² 0.97) and predicts max-values (R² 0.97). But three independent diagnostics — feature-MMD (p=0.002), RQA (p<10⁻⁴³ on every measure), Rosenstein λ₁ (p≈2×10⁻¹⁵) — show generated samples are **quantifiably less chaotic than real**. Rather than report this as a "limitation," we make the diagnostics themselves the contribution: a **three-lens NLD evaluation protocol** sensitive to dynamical-invariance defects that standard recon-quality metrics miss. We then apply it to seven generative architectures (LSTM-VAE family, latent-ODE, Transformer-VAE, KAN-VAE, direct GLV regression) to characterize which inductive biases preserve which dynamical invariants. The deterministic-decoder hypothesis is tested directly via a stochastic-decoder variant. The paper's contribution is the **protocol + the comparative study + the causal demonstration**, not the model.

### 1.2 Roadmap snapshot (live)

| Pivot-era task | Status |
|---|---|
| Design doc written | ✓ `docs/superpowers/specs/2026-05-15-comparative-evaluation-design.md` |
| REFERENCES.md seeded | ✓ |
| Wiki / plan / README updated | ✓ |
| Data pipeline rebuilt w/o sort (D1 fix) | ✓ `data/TRAIN_FINAL_NOSORT.pkl` + `data/TEST_FINAL_NOSORT.pkl` |
| 7 model architectures implemented + smoke-tested | ✓ (`src/models/{cvae,cvae_stochastic,latent_ode,transformer_vae,kan_vae,glv_regression}.py`) |
| Unified eval harness | ✓ `analysis/evaluate_all_models.py` (validated on v1 ckpt) |
| **D3 fix verified on v1 model (§3.8.1)** | ✓ density/coverage 0.13→0.98, MMD p=0.005 survives |
| OOD family test sets (`r~Exp(1)`, `r~Exp(5)`) | ✓ 5k samples each |
| Lens-validation synthetic-perturbation experiment | ✓ `RESULTS_LENS_VALIDATION.json` + figure |
| Model 1 (scale-cond VAE) retrained × 3 seeds | seed 42 ✓ (`model_1_seed42.pth`, 13:19 UTC 2026-05-15); seeds 123, 2026 queued |
| Model 2 (no-cond VAE) retrained × 3 seeds | seed 42 ✓ (`model_2_seed42.pth`, 17:10 UTC 2026-05-15); seeds 123, 2026 queued |
| Model 3 (stochastic-decoder VAE) trained × 3 seeds | seed 42 ✓ (`model_3_seed42.pth`, 21:13 UTC 2026-05-15); seeds 123, 2026 queued |
| Model 4 (Latent-ODE) trained × 3 seeds | seed 42 ✓ (`model_4_seed42.pth`, 00:15 UTC 2026-05-16); seeds 123, 2026 queued |
| Model 5 (Transformer-VAE) trained × 3 seeds | seed 42 ✓ (`model_5_seed42.pth`, 06:57 UTC 2026-05-16, ran 6h42m — heavier than expected); seeds 123, 2026 queued |
| Model 6 (KAN-VAE) trained × 3 seeds | seed 42 training (71% at 18:20 UTC 2026-05-16; 121s/iter now, slowing slightly; ETA ~01:20 Spain Sunday). Loss 0.051, recon 0.008 — converging to similar performance as the other VAEs at much higher compute cost (the early-stage "recon 0.003" reading was a β-warmup artifact, not signal). Seeds 123, 2026 queued |
| Model 7 (Direct GLV regression) trained × 3 seeds | seed 42 ✓ (`model_7_seed42.pth`, 21:22 UTC 2026-05-15, only 9 min); seeds 123, 2026 queued |
| `RESULTS_COMPARATIVE.json` | ✓ seed-42 row populated 00:15 UTC 2026-05-17 (all 7 models). Seed-123 + seed-2026 rows fill as autoqueue progresses |
| Comparative figures | seed-42 versions regenerated 00:15 UTC 2026-05-17 (`fig_comparative_table.{pdf,png}`, `fig_recon_vs_chaos.{pdf,png}`) |
| Phase-4 draft | starts once multi-seed table exists (~Monday) |

### 1.3 Seed-42 comparative findings (added 2026-05-17, headline)

**Single-seed run on all 7 architectures, full eval matrix.** Real data is the 39k-sample no-sort test set; 2k generated per model (extinction-fix θ=0.005), 200 per RQA + Lyapunov. JSON: `RESULTS_COMPARATIVE.json`. Numbers re-derive with `python analysis/evaluate_all_models.py --checkpoints …`.

#### Recon + scale prediction

| Model | recon R² (norm) | recon R² (orig) | max-val R² (pooled) |
|---|---|---|---|
| m1 scale-cond VAE | 0.904 | 0.683 | **0.971** |
| m2 no-cond VAE | 0.922 | 0.217 | **0.047** |
| m3 stochastic VAE | 0.904 | 0.682 | 0.972 |
| m4 Latent-ODE | 0.927 | 0.682 | 0.975 |
| m5 Transformer-VAE | 0.936 | 0.675 | 0.962 |
| m6 KAN-VAE | 0.907 | 0.601 | 0.805 |
| m7 GLV-regression | n/a (generative-only) | n/a | n/a |

**Reading.** The v1 scale-conditioning result generalizes cleanly to the no-sort pipeline and **across architectures**: every scale-conditioned model lands max-val R² in [0.96, 0.98]; the no-cond ablation collapses to **0.047** (worse than even the v1 baseline's 0.59, which had the sort step helping it). KAN-VAE underperforms on max-value (0.805) without obvious upside — KAN's function-approximation basis is not paying off for scale prediction on this task. Transformer-VAE has the best *normalized* recon (0.936); LSTM-VAEs cluster around 0.90.

#### Distributional fidelity (Lens 1)

| Model | MMD² (lower = better) | density@5 | coverage@5 |
|---|---|---|---|
| m1 scale-cond VAE | 6.10e-02 | 0.731 | 0.652 |
| m2 no-cond VAE | 1.26e-01 | 0.395 | 0.467 |
| m3 stochastic VAE | 4.83e-02 | 0.759 | 0.633 |
| m4 Latent-ODE | 5.64e-02 | 0.486 | 0.621 |
| m5 Transformer-VAE | 3.00e-02 | 0.769 | 0.691 |
| m6 KAN-VAE | 4.32e-02 | 0.617 | 0.637 |
| **m7 GLV-regression** | **1.00e-02** | 0.764 | **0.847** |

**Reading — the most paper-disruptive finding.** The "physics-naive" inverse-problem baseline **wins distributional fidelity decisively**: MMD² = 0.010 (3× better than Transformer-VAE, 12× better than the no-cond ablation), coverage 0.847 (Transformer-VAE second at 0.69). "Just regress the ODE parameters and integrate" produces samples that match real most closely in feature space. The fancier generative ML doesn't lose by a lot, but it doesn't beat the physics-informed baseline. This re-frames the paper's "what does generative ML buy you over inverse-problem-plus-integration?" question into a real comparison rather than rhetorical setup.

#### NLD invariants (Lens 2 + Lens 3) — the architecture-independent finding

For every model, real RQA-DET = 0.617 (unchanged — same real-data subsample). Generated DET clusters tightly:

| Model | gen RQA-DET | DET KS p | gen λ₁ | λ₁ KS p |
|---|---|---|---|---|
| m1 scale-cond VAE | 0.991 | 1.4e-88 | +0.052 | 6.7e-06 |
| m2 no-cond VAE | 0.991 | 1.4e-88 | +0.053 | 4.6e-07 |
| m3 stochastic VAE | **0.987** | 9.5e-79 | **+0.054** | 1.8e-04 |
| m4 Latent-ODE | 0.990 | 9.8e-85 | +0.052 | 1.4e-06 |
| m5 Transformer-VAE | 0.986 | 2.2e-75 | +0.053 | 6.7e-06 |
| m6 KAN-VAE | 0.991 | 1.4e-88 | +0.051 | 8.1e-07 |
| m7 GLV-regression | 0.988 | 9.5e-79 | +0.058 | 4.3e-04 |

(Real λ₁ = +0.076 across the board.)

**Reading.** Every model — *including the Latent-ODE with its continuous-time prior*, the *stochastic-decoder variant that tested the determinism hypothesis directly*, and the *GLV-regression baseline that literally integrates the ODE* — produces RQA-DET in [0.986, 0.991]. The gap of 0.37 vs real is **identical to within 0.005 across all 7 architectures**. Same story for λ₁: every model produces +0.05-0.06, real is +0.076, gap ~0.02-0.025 with the same KS p-value order of magnitude.

This is a **paper-positive finding stated honestly**: the smoother-than-real phenomenon the protocol detects **is not architecture-specific**. It survives:
- Architectural diversity (recurrent / attention / continuous-time / KAN basis).
- Decoder stochasticity (m3 vs m1: ΔDET = 0.004, basically zero).
- The "just integrate the ODE" approach (m7).

What this means for the paper's framing:

1. **The 3-lens protocol is sensitive to a real, persistent feature of the generated-trajectory distribution that no architectural choice has fixed.** That is exactly what an *evaluation* protocol should do — detect a property invisible to recon-R².
2. **The deterministic-decoder hypothesis test (m3) is INVALID — see §1.3.1 below.** ⚠️ The learned σ on m3's stochastic decoder converged to σ ≈ 0.00044 (essentially zero) — the optimizer drove it down because the MSE recon loss penalizes any noise. So m3 effectively became a deterministic decoder during training and the "m3 vs m1 ≈ identical" comparison is uninformative for the hypothesis. **Retraining m3 with frozen σ ∈ {0.05, 0.1, 0.2}** is now scheduled as part of the spectral-loss / decoder-noise experimental batch.
3. **The "comparison reveals architectural trade-offs" arc that motivated the pivot is *less interesting than expected on this single seed***. All 7 architectures cluster tightly on the NLD metrics. The architectural diversity is real on max-val R² (m1=0.97, m2=0.05) and distributional fidelity (m7 best, m2 worst), but the chaos diagnostics show essentially one cluster.
4. **The headline pivots, again, toward a method paper.** The 3-lens protocol detects a property that recon-R² and even Lens 1 (MMD on coarse features) miss, and that property is *invariant* across the architectures you can plausibly use for this problem. That's a strong sales pitch for the protocol; it's a less-strong sales pitch for any particular model.

Wait for seed-123 and seed-2026 to confirm the architecture-invariance is not a seed artifact. If two more seeds show the same pattern, this is the paper's headline.

Source: `RESULTS_COMPARATIVE.json` (50 KB), `final figures/fig_comparative_table.{pdf,png}`, `final figures/fig_recon_vs_chaos.{pdf,png}`.

#### 1.3.1 Pre-multi-seed validation sweep (2026-05-17, before seeds 123/2026 finish)

Five sanity / extension checks before committing the ~40 GPU-hr remaining multi-seed compute:

| Check | Result | Wiki impact |
|---|---|---|
| **A1**: protocol on original-scale vs normalized real data | Real DET = 0.59 (orig) vs 0.65 (norm); λ₁ = +0.079 in both. The "real 0.6, gen 0.99" gap is **NOT a normalization artifact** — it reflects a genuine generator defect | §1.3 RQA finding is robust |
| **A2**: visual inspection of generated samples from all 7 models | Strong qualitative confirmation. Real samples show multi-peak oscillation; m1/m3/m4/m5/m6 produce fast transient + monotonic settling (mode-covering / exponential decay); m2 is even flatter; **m7 GLV-regression visibly oscillates** — consistent with its lens-1 win. Figure: `final figures/fig_visual_sanity_seed42.{pdf,png}` | New figure for paper Section 3, "what 'smoother than real' actually looks like" |
| **A3**: each ckpt produces non-garbage forward passes | ✓ all 7 |  — |
| **B3**: inspect m3 stochastic-decoder learned σ | ⚠️ **σ = 0.00044, essentially zero.** Optimizer drove it down because MSE recon term penalizes any noise added to the output. **The "m3 vs m1 → decoder stochasticity is insufficient" conclusion is invalid** — m3 effectively trained as a deterministic decoder. The hypothesis is *untested*, not *disproved*. | Edit §1.3 finding #2: scheduled retrain with frozen σ ∈ {0.05, 0.1, 0.2} as part of the B1 batch |
| **B2**: OOD eval on Exp(1) and Exp(5) test sets | in progress (results land in `RESULTS_COMPARATIVE_OOD_Exp{1,5}.json`) |  — |
| **B1**: spectral-loss VAE variant + frozen-σ variants | designed, queued to run on free GPU window |  — |

The A1 and A2 results substantially **strengthen** the paper's headline (the protocol works, the defect is visible). The B3 result substantially **weakens** one of the four supporting claims (decoder stochasticity → null) and forces a real retrain to settle the hypothesis. Net: better paper after the validation than before.

#### 1.3.2 OOD Exp(1) result — major reframe of the §1.3 "architecture-invariant defect" story (2026-05-17)

**Ran the unified eval on all 7 seed-42 checkpoints against `data/TEST_OOD_Exp1.pkl`** (5000 trajectories generated with `r ~ Exp(1)` — faster growth rates than the training distribution Exp(2)).

| Model | recon R² | max-val R² | MMD² (lower=better) | DET KS p (higher=match) | λ₁ KS p |
|---|---|---|---|---|---|
| m1 scale-cond | 0.877 | 0.972 | 1.6e-01 | **0.39** | 1.2e-02 |
| m2 no-cond | 0.911 | -0.167 | 7.0e-02 | 1.6e-02 | 5.2e-02 |
| m3 stochastic | 0.879 | 0.972 | 1.7e-01 | 6.1e-03 | **0.11** |
| m4 Latent-ODE | 0.929 | 0.974 | 1.0e-01 | 4.3e-03 | **0.11** |
| m5 Transformer | 0.939 | 0.965 | 1.7e-01 | 2.4e-06 | 3.0e-02 |
| m6 KAN | 0.864 | 0.818 | 1.1e-01 | **0.11** | 8.5e-03 |
| m7 GLV-regression | n/a | n/a | 6.5e-02 | 4.3e-04 | 4.0e-02 |

**Real RQA-DET on Exp(1) = 0.993** (vs Exp(2) where it was 0.617). Generated DET clusters in [0.985, 0.992]. **Gap ≤ 0.007 across all 7 models.** Several KS p-values are non-significant (m1, m3, m4, m6 on at least one of DET or λ₁).

**The reframe.** On the ID test set Exp(2), real DET = 0.617 and gen DET = 0.99 → gap 0.37, p < 10⁻⁷⁵. On Exp(1), real DET = 0.993 and gen DET = 0.99 → gap < 0.01, often non-significant. **The "smoother-than-real" defect vanishes when we change the OOD regime.** This is NOT random — it's structural:

1. **Generators are regime-locked.** They learned to produce trajectories with DET ≈ 0.99 from the training data, and they do so regardless of which test set you compare to. This is the inductive bias of MSE-trained autoregressive decoders on this preprocessing pipeline.
2. **The protocol does its job honestly.** On Exp(2) it correctly flags mismatch; on Exp(1) it correctly flags match. This is exactly how a discriminative test should behave.
3. **The framing changes from "architectures fail at chaos" to "architectures have a strong inductive bias toward smooth trajectories, which matches some regimes and not others."** That's a more publishable claim and a richer paper Section 5.

For the paper, this OOD finding becomes a load-bearing piece:
- It demonstrates the protocol's *specificity* (not just *sensitivity*).
- It explains *why* all architectures cluster: they all converge to the same MSE-driven mode-covering equilibrium.
- It opens a clean section on "characterizing the regimes where current generative ML works on dynamical systems."

Waiting on OOD Exp(5) eval (slower growth than Exp(2)) — if Exp(5) shows real-DET < 0.6 and the generators still produce DET ≈ 0.99, the regime-locked-generator interpretation is **confirmed** in both OOD directions. That would be a Section-5 figure: real-DET (x) vs gen-DET (y) across (Exp(1), Exp(2), Exp(5)) regimes, generators sitting on a flat horizontal line at 0.99 regardless of x.

Source: `RESULTS_COMPARATIVE_OOD_Exp1.json`. Exp(5) eval running in background (PID 1081971).

#### 1.3.3 OOD Exp(5) result — story sharpens further (2026-05-17, completed 07:36 UTC)

**Predicted:** Exp(5) (slower growth, expected more variable trajectories) would have real-DET ≈ 0.5, completing the U-shape (Exp(1)=0.99, Exp(2)=0.62, Exp(5)=0.5) and confirming "generators are regime-locked at 0.99 independent of real-data DET."

**Actual:** Exp(5) real-DET = **0.993** — same as Exp(1). It's the *training distribution Exp(2)* that's the outlier at 0.617, NOT the OOD regimes.

This **deepens** the story rather than confirming the simpler regime-lock interpretation:

| Distribution | Real DET | Gen DET (range across 7 models) | Gap |
|---|---|---|---|
| Exp(1) — OOD, faster growth | 0.993 | [0.985, 0.992] | 0.001–0.008 (often non-significant) |
| **Exp(2) — TRAINING distribution** | **0.617** | [0.986, 0.991] | **0.37 (always p < 10⁻⁷⁵)** |
| Exp(5) — OOD, slower growth | 0.993 | [0.985, 0.991] | 0.002–0.008 (significant but small) |

| Distribution | Real λ₁ | Gen λ₁ (range) | Match? |
|---|---|---|---|
| Exp(1) | +0.058 | [+0.043, +0.061] | mostly good |
| **Exp(2)** | +0.076 | [+0.051, +0.058] | gap ~0.02, all p < 10⁻⁵ |
| Exp(5) | +0.048 | [+0.044, +0.069] | mixed — m1 best (0.044 vs 0.048), others overshoot |

**The honest claim emerges as:** ***MSE-trained autoregressive decoders, trained on the most variable subset of GLV systems, generate trajectories that are smoother than the training data they were trained on.*** They converge to producing DET ≈ 0.99 regardless of where they're tested. This is *not* "regime-lock" (which would mean they produce 0.99 because that's what they saw); it's **mode-covering toward an inductive-bias attractor that differs from the training distribution itself**.

Why this is paper-good (much better than the original §1.3 framing):

1. **The protocol's sensitivity matches the data's variability.** On Exp(2) where real is variable, the protocol catches mismatch; on Exp(1)/Exp(5) where real is also high-DET, mismatch is small. That validates the protocol as a *measurement* tool.
2. **The generator failure is sharper:** they don't produce what they were trained on. The training data has DET 0.62 (variable); the generated data has DET 0.99 (smooth). That's a *real loss of training-distribution coverage*, not a mismatch with some held-out unfamiliar regime.
3. **The "and here's the cure" question now has teeth.** The spectral-loss + frozen-σ experiments (Phase B1) ask: can we change the training objective to make generators produce DET = 0.62 like their training data, rather than 0.99? If yes, we have diagnosis + cure in one paper. If no, the inductive-bias attractor is genuinely robust to surface-level interventions.
4. **m7 GLV-regression behaves differently** — on Exp(5) its λ₁ overshoots dramatically (+0.069 vs real +0.048). This is consistent with the inverse-problem solver predicting (r, A) from a short trajectory and then integrating: if it gets r̂ slightly too high, the trajectory blows up. Real Exp(5) has *smaller* growth rates than the inverse-solver was trained on (it was trained on Exp(2)). m7 has a different failure mode — *overshooting* — that the protocol cleanly distinguishes.

**For the paper's Section 5:** a 3-panel figure (one per OOD distribution) of real-vs-gen DET histograms, or a single scatter of (real-DET, gen-DET) across all three regimes with the "generators sit at 0.99" line visually striking. This is now the paper's centerpiece comparative result.

Source: `RESULTS_COMPARATIVE_OOD_Exp5.json` + `RESULTS_COMPARATIVE_OOD_Exp1.json` + `RESULTS_COMPARATIVE.json`.

---

## 2. What We Built

### 2.1 Data pipeline (`data_generation/`)

- **GLV generator** (`generate_family_FIXED.py`, `custom_glv_FIXED.py`): produces 7-species systems with `r ~ Exp(2)`, $A_{ii} \sim -\mathrm{Exp}(2)$, off-diagonals $\mathcal{N}(0,1)$, eigenvalue stability check, positive fixed point, `scipy.integrate.solve_ivp` integration, 30-second timeout per seed, downsample 129→65 points. Train/test seeds are separated by >800M (no possible collision).
- **3-stage preprocessor** (`preprocessor.py`):
  1. **Family normalization**: divide each sample by `max` over all (species, time). Highest peak in the family = 1.
  2. **Sort curves by peak** (descending) so curve 0 is always the dominant one.
  3. **Per-curve normalization**: each curve's own peak = 1.

  The two scale factors (`family_max_values`, `reconstruction_max_values`) are stored alongside the normalized tensor; the original trajectory is recoverable as `data * reconstruction_max_values * family_max_values`.

- **Dataset sizes** (verified 2026-05-14 by loading the pickles): TRAIN = **117,472** samples; TEST = **39,189** samples; both at shape `(N, 7, 65)`. Older docs (`CODEBASE_REFERENCE.md`) quote 156,800 — that was the *generation target*, the actual saved file is smaller after quality rejection.
- **Parameter-recovery matched set** (added 2026-05-14): an additional 10,000 trajectories generated with seed base `555_000_001` (outside train/test seed ranges) where `(r, A)` are also recorded. Stored in `data/PARAM_RECOVERY_MATCHED.pkl`; used in the §4.5 experiment.

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

## 3. Verified Performance — v1 single-model results (with-sort preprocessing)

> **All numbers in §§3–4 below are for the v1 model (single scale-conditioned VAE, with-sort preprocessing).** They informed the 2026-05-15 pivot but are *not* the final paper's headline results. The comparative table across the 7 architectures will live in a new §5 once training completes. Existing v1 numbers are kept here in full because (a) they were verified, (b) they tell the story that motivated the pivot, and (c) the comparative paper still cites the scale-conditioned VAE as one of the seven models — these are its numbers, on the old preprocessing. **Numbers re-derived on the no-sort pipeline may shift; the qualitative findings about chaos-under-modeling are expected to persist.**

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

Canonical numbers from `RESULTS.json` (N=2000, bootstrap 95% CIs):

| Source | Mean LV-$R^2$ | 95% CI | Median | % > 0.9 | % > 0.95 |
|---|---|---|---|---|---|
| **Real test data** | **0.9875** | [0.9869, 0.9881] | 0.9913 | **99.7%** | 97.8% |
| Generated (raw) | 0.9588 | [0.9560, 0.9615] | 0.9868 | 85.7% | 76.8% |
| Generated (extinction θ=0.005) | 0.9680 | [0.9660, 0.9701] | 0.9876 | 91.5% | 80.5% |

Generated samples are LV-consistent (~97% mean R², 92% are >0.9 with the extinction fix), but they do **not** exceed real data. The buggy "gen > real" summary has been deleted. The honest story — "generated samples adhere to LV equations at ~97% R² with the extinction fix, within 0.02 of real data, with the gap explained by occasional near-extinction trajectories" — is the result we report.

Saved to `RESULTS.json` (regenerable via `analysis/produce_paper_metrics.py`).

### 3.5 Novelty / memorization check

Nearest-neighbor distances (Euclidean over flattened normalized curves), 1k generated samples vs 20k random train/test samples each:

| Distance | Mean |
|---|---|
| Generated → nearest train | 3.43 |
| Generated → nearest test | 3.44 |
| Test → nearest train (baseline) | 3.62 |
| Generated → nearest other generated | 3.88 |
| Memorization ratio (gen→train / test→train) | **0.946** |

Generated samples sit *slightly closer* to the training set than held-out test samples do (ratio 0.946). The model is not copying — internal distances among generated samples are larger than gen-to-train — but the latent prior is somewhat biased toward dense training regions. This was a yellow flag; the proper statistical follow-up (MMD permutation test + density/coverage on dynamical-feature vectors) has been completed and is reported in §3.8. Source: `RESULTS.json`.

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

### 3.7 Baseline comparison: conditioned vs non-conditioned (added 2026-05-14)

Trained a non-conditioned 30D LSTM-VAE (`model_ckpts/model_final_30_baseline.pth`) with `use_scale_conditioning=False` and otherwise identical architecture, data, and training schedule (1000 epochs, β warmup 300, TF decay 400, λ_maxval = 0.5). This isolates the contribution of the scale-conditioning architectural change.

| Metric | Conditioned | Baseline (no cond.) | Δ |
|---|---|---|---|
| Recon $R^2$ (normalized) | 0.9335 | **0.9449** | −0.011 |
| Recon $R^2$ (original scale) | **0.9654** | 0.8444 | **+0.121** |
| Recon MAE (normalized) | 0.0541 | 0.0490 | +0.005 |
| Max-value $R^2$ (curves 1–6) | **0.9711** | 0.5905 | **+0.381** |

**Per-curve max-value $R^2$** (the critical one):

| | x1 | x2 | x3 | x4 | x5 | x6 |
|---|---|---|---|---|---|---|
| Conditioned | 0.946 | 0.946 | 0.928 | 0.907 | 0.869 | 0.783 |
| Baseline    | −0.308 | −0.289 | −0.226 | −0.146 | −0.043 | 0.036 |

The baseline produces **negative** $R^2$ on 5 of 6 max-value targets — worse than predicting the mean, exactly as expected: it has no scale input, so the predictor head can only memorize the marginal distribution.

| Latent metric | Conditioned | Baseline |
|---|---|---|
| Active dims (var ≥ 0.01) | 25 / 30 | **30 / 30** |
| Collapsed dims | 5 / 30 | 0 / 30 |
| PCs for 90% / 95% / 99% var | 20 / 22 / 23 | 25 / 27 / 30 |

**Interpretation:**

1. **Max-value prediction**: the conditioned model is decisively better (+0.38 absolute $R^2$). The baseline confirms that scale conditioning is *necessary* — without it, max-value prediction is broken.
2. **Reconstruction $R^2$ on normalized space**: baseline is *marginally* better (+0.011). This is the expected trade-off — the conditioned model spends part of its capacity learning scale, leaving slightly less for shape; the gain comes back many-fold on the original-scale metric.
3. **Reconstruction on original scale**: conditioned wins by +0.12, because the baseline's broken max-value prediction destroys the denormalization.
4. **Latent geometry**: baseline shows zero posterior collapse but uses all 30 dims, suggesting the additional capacity goes to (futile) attempts at scale recovery from normalized data. The conditioned model is more efficient (25 active dims, 20 PCs for 90% variance).

Numbers from `RESULTS_BASELINE.json` and `RESULTS_COMPARISON.md` (the canonical paper Table 1).

### 3.8 Novelty / coverage — statistical test (added 2026-05-14)

The §3.5 nearest-neighbor distance gave a single number (memorization ratio 0.95) that hinted at distributional mismatch without quantifying it. `analysis/novelty_coverage.py` runs a proper two-sample test on a **26-D dynamical-feature vector** (per-species mean/std/trend + 5 global features: total variance, mean correlation, mean extrema count, mean curvature, mean dominant frequency). N = 2000 per group. Generated samples are post-processed with the extinction threshold θ=0.005.

| Test | Value |
|---|---|
| **MMD² (Gaussian kernel, median heuristic)** | 0.0677 |
| Permutation null mean ± std | 0.0000 ± 0.0002 |
| Permutation p-value (n_perm = 500) | **0.0020** |
| Density@k=5 (Naeem 2020) | 0.135 |
| Coverage@k=5 | 0.246 |
| Density (gen → real, swapped) | 0.075 |
| Coverage (swapped) | 0.209 |
| Features distinguishing real vs gen (KS p<0.05) | **20 / 26** |

**Verdict (honest):** real and generated samples are *statistically distinguishable* in dynamical-feature space (p ≈ 0.002). Coverage of 0.25 means only ~25% of real test points have a generated sample within their 5-NN ball — generated samples form a thinner/shifted distribution.

**Where does the mismatch come from? The KS test points the finger:**

| Feature | KS statistic | What it means |
|---|---|---|
| `mean_extrema` | **0.983** | Generated trajectories have a very different peak/trough count from real |
| `mean_curvature` | **0.685** | Generated trajectories are markedly less "bumpy" (smoother) |
| `sp3_std`, `sp6_std`, `sp6_mean`, `sp6_trend` | 0.09–0.13 | Subdominant-species statistics differ substantially |
| Species means (sp0–sp5) | 0.02–0.06 | Mostly match real |
| Dominant frequency | 0.025 | Matches real |

**Interpretation:** the model generates *smoother, less oscillatory* trajectories than real data. Variances and extrema counts are mismatched, but means are roughly right. This is classic mode-covering behavior in autoregressive VAEs — the decoder regresses to the conditional mean and loses high-frequency content. The mismatch is most severe in subdominant species, which is also where parameter recoverability fails (§4.5). Both observations point to the same limitation: **the model captures dominant-species macroscopic dynamics well and misses fine-grained variability**.

**For the paper:** report this honestly as a limitation. The fix is well-known in the literature (autoregressive decoder with output noise, or a perceptual-style loss that penalizes spectral mismatch) and goes into future work. This finding does not undercut the architectural contribution; it characterizes its scope.

Source: `RESULTS_NOVELTY.json`, `final figures/fig_novelty_coverage.pdf`.

#### 3.8.1 D3 fix (2026-05-15, pivot): cleaner feature set, finding holds

The §3.8 v1 finding rested on a 26-D feature vector including `mean_extrema` (KS 0.98 — driving the headline) and `mean_curvature` (KS 0.69). Both are integer-valued or near-integer; the first is essentially a peak count, the second a smoothed version of the same. Re-running the analysis on the **24-D feature vector with these two dropped** (via `analysis/evaluate_all_models.py`):

| Metric | v1 (26-D, with sort) | D3-corrected (24-D, same v1 model) |
|---|---|---|
| MMD² (observed) | 0.0677 | 0.0035 |
| Permutation p | 0.002 | **0.005** |
| Density@5 | 0.135 | 0.983 |
| Coverage@5 | 0.246 | 0.948 |
| Features distinguishing real vs gen (KS p<0.05) | 20 / 26 | **18 / 24** |

Two things change:
1. **The MMD finding survives the D3 fix.** Real and generated are still statistically distinguishable in feature space (p = 0.005). The smoother-than-real story is not an artifact of the integer-valued features.
2. **Density/coverage rise dramatically** (0.13/0.25 → 0.98/0.95). The collapse in v1 was driven by the integer-valued features inflating distances. **With clean features, the model actually *covers* the real distribution well at the k-NN scale.** The mismatch is concentrated in *fine-grained NLD invariants* — picked up by Lens 2 (RQA) and Lens 3 (Lyapunov) but invisible at the coarse k-NN scale.

This is a paper-positive finding: the model's distributional fidelity is much better than v1 reported on coarse metrics, and the mismatch sits specifically in dynamical-invariance properties — which is exactly what the 3-lens protocol is designed to detect. Reframes §3.8 from "the model is statistically distinguishable from real" to "the model matches real *coarse-grainedly*, but the NLD-invariance protocol detects fine-grained differences invisible to standard sample-quality metrics" — a much sharper statement.

Source: `RESULTS_COMPARATIVE_v1_sanity.json`, run 2026-05-15.

### 3.9 Chaos diagnostics — RQA and largest Lyapunov exponent (added 2026-05-15)

CSF reviewers will expect at least one nonlinear-dynamics analysis beyond the recurrence plots we already have. `analysis/chaos_diagnostics.py` computes two standard NLD diagnostics on the species-averaged signal of matched real vs generated samples (n = 200 per group; generated samples post-processed with the extinction fix θ = 0.005). All measures are computed from a Takens embedding with m = 3, τ = 2 (so the recurrence matrix has N = 61, sufficient for distributional comparison; per-trajectory estimates of λ₁ are noisy at T = 65 and we declare that as a limitation).

**Recurrence Quantification Analysis** — ε is chosen per trajectory to fix the recurrence rate at 10 % so RR matches between groups by construction; the discriminative content lives in DET, L_mean, L_max, LAM, TT.

| Measure | Real (mean ± std) | Generated (mean ± std) | KS stat | KS p-value |
|---|---|---|---|---|
| Recurrence rate RR (target) | 0.100 ± 0.000 | 0.100 ± 0.000 | 0.00 | 1.0 |
| **Determinism DET** | **0.588 ± 0.244** | **0.990 ± 0.011** | 0.95 | 2 × 10⁻⁹⁸ |
| Mean diagonal line L_mean | 5.46 ± 2.88 | 13.30 ± 4.99 | 0.76 | 7 × 10⁻⁵⁷ |
| Max diagonal line L_max | 16.7 ± 11.8 | 35.4 ± 7.3 | 0.72 | 2 × 10⁻⁵⁰ |
| Laminarity LAM | 0.680 ± 0.194 | 0.960 ± 0.027 | 0.85 | 3 × 10⁻⁷³ |
| Trapping time TT | 3.74 ± 1.15 | 5.87 ± 1.26 | 0.67 | 7 × 10⁻⁴³ |

**Largest Lyapunov exponent** (Rosenstein, fit on first 50 % of the divergence curve):

| | Real | Generated | KS stat | KS p-value |
|---|---|---|---|---|
| λ₁ (per timestep) | **+0.079 ± 0.039** | **+0.031 ± 0.064** | 0.41 | 2 × 10⁻¹⁵ |

**Interpretation:** all five non-trivial RQA measures and the Lyapunov exponent differ significantly between real and generated populations, and they tell *one coherent story*: generated trajectories are **more deterministic, more laminar, and less chaotic** than real ones (DET 0.99 vs 0.59, LAM 0.96 vs 0.68, λ₁ +0.03 vs +0.08). This is exactly the same finding as §3.8 (real and generated distinguishable in feature space, gap concentrated in `mean_extrema` and `mean_curvature`) — RQA and Lyapunov put a hard NLD number on the *smoother-than-real* limitation.

**Why this finding is paper-positive, not paper-negative:**

1. It is *consistent across independent metrics* — the same defect surfaces in the dynamical-feature KS test (§3.8), in the parameter-recoverability ceiling on cross-species coupling (§4.5), and in the chaos diagnostics here. Three independent lenses point at one phenomenon.
2. It is *the textbook mode-covering signature of an autoregressive VAE with a deterministic decoder* — the model regresses to the conditional mean and discards high-frequency content. The literature has well-known fixes (decoder stochasticity, perceptual / spectral loss, autoregressive output noise). We report it as a clean limitation and direct future-work bullet, not as a surprise.
3. Real-data λ₁ is *not* near zero — the real GLV systems have positive Lyapunov on this embedding, which is a non-trivial sanity check that the diagnostic is sensitive to chaotic content. Generated samples have λ₁ closer to zero, which is what "smoother" looks like in NLD terms.

Source: `RESULTS_CHAOS.json`, `final figures/fig_chaos_diagnostics.pdf`.

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

## 4.5 Parameter Recoverability — `μ(z) → (r, A)` (added 2026-05-14)

To answer the "what does the model know about the underlying physics?" question quantitatively, we generated a fresh held-out dataset of **N = 10,000 trajectories** with seed base `555_000_001` (outside both train and test seed ranges) where we also recorded the GLV parameters `(r, A)` used to integrate each trajectory. We preprocess identically to training (family-norm → sort-by-peak → per-curve norm), encode through the trained 30D conditioned CVAE to get `μ(z) ∈ ℝ³⁰`, *permute `r` and `A` row/column to match the same sort the model sees*, and fit Ridge regression with 5-fold CV.

Sanity check: reconstruction R² on this fresh dataset = **0.939** — matches the test-set 0.93, so the matched data is in-distribution and the encoder produces meaningful `μ(z)`.

**Out-of-fold Ridge R² (5-fold CV):**

| Target | Mean $R^2$ |
|---|---|
| Growth rate **r** (7 outputs) | **0.241** |
| Interaction matrix **A** (49 outputs) | 0.052 |
|   ↳ Diagonal of A (self-regulation) | **0.195** |
|   ↳ Off-diagonal of A (cross-species) | 0.028 |
| Re(eig A) (system stability) | 0.162 |
| Im(eig A) (oscillation frequency) | 0.014 |

**Per-species growth rate $R^2$** (species 0 = highest peak):
0.43, 0.38, 0.32, 0.06, 0.19, 0.11, 0.20.

**Top-5 most-recoverable A entries** are all diagonal: $A_{00}$ (0.39), $A_{11}$ (0.29), $A_{22}$ (0.24), $A_{44}$ (0.15), $A_{10}$ (0.14). **Bottom entries** are all far off-diagonal entries involving the smallest species: $R^2$ around zero.

### What this means

This is a **partial-recoverability** result, and it's *informative* — three concrete claims emerge:

1. **The latent space recovers what shapes the dominant-species trajectory.** The growth rate of the most-dominant species is the best-recovered target ($R^2 = 0.43$); subdominant species drop off. The same is true for the diagonal of $A$: the model knows the self-regulation of the species you can *see* clearly, not the ones near the noise floor.
2. **Cross-species coupling is essentially invisible to the latent space.** $A_{ij}$ for $i \ne j$ has mean $R^2 = 0.03$. This is consistent with what we already knew: only ~7% of latent dimensions encode interaction-style features (Dim 2 ↔ synchronization, $r = 0.38$) — too thin to back out individual coupling coefficients.
3. **System stability is partially encoded; oscillation frequency is not.** Real parts of the eigenvalues of $A$ are recoverable at $R^2 = 0.16$; imaginary parts at $R^2 = 0.01$. The model captures "is this regime stable?" better than "at what frequency does it oscillate?"

### Honest framing for the paper

The pre-experiment narrative we were *tempted* to write was "the model implicitly identifies GLV parameters." The honest result is **the model identifies the parameters of the species you can see, and is blind to the rest**. That sentence is publishable as-is. It dovetails with the species-centric interpretability story (§4) and explains *why* the latent space looks the way it does: the normalization-and-sort pipeline preserves dominant-species information by construction and discards the rest.

This is the *new headline figure of the paper* (`final figures/fig_param_recoverability.pdf`).

Stored:
- `data/PARAM_RECOVERY_MATCHED.pkl` — 10k trajectories + `(r, A)` + sort permutations (78 MB)
- `RESULTS_PARAM_RECOVERY.json` — all numbers above
- `analysis/parameter_recoverability.py` — reproducible from scratch (~6 min generation + ~1 min eval)
- `final figures/fig_param_recoverability.{pdf,png}` — the figure

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
| `fig_chaos_diagnostics.pdf` | RQA distributions (DET, L_mean, L_max, LAM, TT) + λ₁ histogram, real vs generated (§3.9) |
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

- ✅ Old datasets pruned 2026-05-15: `data/` dropped from 4.5 GB → 1.8 GB; only the live pipeline files (FIXED, NOSORT, PROCESSED, OOD, PARAM_RECOVERY) remain. Legacy `checkpoints/` dir also deleted.
- ✅ wandb pruned 2026-05-15: 187 runs >1 year old deleted (~1.1 GB); pivot-era wandb dir now routed to `/mnt/storage/shared/TimeSeries/wandb` so root disk doesn't fill.
- **Methods LaTeX** (`METHODOLOGY_DOCUMENT.md`) describes the v1 non-conditioned 30D and is now doubly stale (the pivot changes the paper's identity entirely); the Phase-4 draft will rewrite methods from scratch against the design doc, not against this file.
- Several `generate_*.py` scripts at the root still overlap and could be moved into `analysis/`; left in place for now since they were used to produce the v1 figures and any reshuffle risks regressions before submission.

---

## 8. What's Solid, What's Soft

### Solid (paper-ready)

- Architecture and training pipeline.
- Reconstruction quality: $R^2 = 0.93$ normalized, $0.97$ denormalized, all curves > 0.91; bootstrap CIs in `RESULTS.json`.
- Max-value prediction: $R^2 = 0.971$, 95% CI [0.9711, 0.9715] — strong architectural contribution.
- Latent-space health: 25/30 active dims, 20 PCs for 90% variance.
- Latent-space interpretability: hierarchical species/global/interaction decomposition with named dim↔feature correlations.
- 50D → 30D ablation: clear and well-documented.
- Extinction-threshold post-processing: principled, with a sweep (θ=0.005 winner).
- **LV adherence (corrected)**: real = 0.99 median, gen with fix = 0.99 median, gap of 0.013 in mean — generated samples are LV-consistent.
- **Parameter recoverability (NEW)**: $μ(z)$ recovers growth rates partially ($R^2 = 0.24$, best species = 0.43), diagonal of $A$ partially ($R^2 = 0.20$), off-diagonal $A$ essentially not at all ($R^2 = 0.03$). Honest framing: *the model identifies parameters of dominant species and is blind to fine cross-species coupling.* See §4.5.
- **Baseline comparison (NEW)**: max-value $R^2$ **conditioned 0.97 vs baseline 0.59**; the architectural contribution is now isolated and decisive (see §3.7).
- **Novelty / coverage statistical test (NEW)**: MMD permutation test p = 0.002 → real and generated distinguishable in feature space. Diagnosed: generated trajectories are *smoother / less oscillatory* than real. Concrete limitation, concrete future-work direction (see §3.8).
- **Chaos diagnostics (NEW)**: RQA (DET, L_mean, L_max, LAM, TT) + Rosenstein's largest Lyapunov, on n = 200 matched real vs generated samples. Every non-trivial RQA measure differs at p < 10⁻⁴³; λ₁ differs at p ≈ 2 × 10⁻¹⁵. Generated trajectories are *more deterministic, more laminar, and less chaotic* than real — same story as §3.8, sharper number. See §3.9.

### Soft (need work before submission)

- **Methods text in LaTeX is out of date** with the conditioned architecture and the corrected numbers (Phase 4 in PLAN.md).
- **Out-of-distribution generation**: `explore_oscillation_extrapolation.py` is half-finished. Either finish it or remove the claim.

### Resolved (was "soft", now solid)

- LV validation narrative ✓ corrected and locked into `RESULTS.json`.
- Confidence intervals ✓ bootstrap CIs added.
- Parameter recoverability question ✓ answered — partial recovery, dominant-species story (§4.5).
- Baseline comparison ✓ done — scale conditioning is decisive (+0.38 max-val $R^2$, see §3.7).
- Novelty / coverage ✓ done — MMD permutation test (p=0.002) shows real and generated *are* distributionally different in feature space, with the gap concentrated in oscillation-related features. Reported honestly as a limitation; concrete future-work direction (§3.8).
- Chaos diagnostics ✓ done — RQA + Rosenstein's largest Lyapunov on n=200 matched real vs generated. All RQA measures differ at p < 10⁻⁴³; λ₁ at p ≈ 2 × 10⁻¹⁵. Generated trajectories are quantifiably *less chaotic* (DET 0.99 vs 0.59, λ₁ +0.03 vs +0.08) — the same smoother-than-real story as §3.8, now anchored in standard NLD measures (§3.9).

---

## 9. Reproduction

```bash
source TimeSeries/bin/activate           # project venv
python train_cvae.py                     # ~2–3 hr on a single GPU, 2000 epochs
# evaluation
python -c "from src.models.cvae import LSTM_VAE; ..."   # see METRICS_VERIFIED.json
```

Verified metrics live in **`RESULTS.json`** + **`RESULTS.md`**, regenerated by `analysis/produce_paper_metrics.py` — recon, max-val, latent health, LV adherence (real vs generated raw vs generated with extinction fix), nearest-neighbor novelty/memorization. Bootstrap 95% CIs included.

The Phase-2 / Phase-3 experiments have their own self-contained scripts and JSON outputs (each can be re-run independently in 1–10 minutes on a single GPU):

- `analysis/parameter_recoverability.py` → `RESULTS_PARAM_RECOVERY.json` (§4.5)
- `analysis/evaluate_baseline.py` → `RESULTS_BASELINE.json` + `RESULTS_COMPARISON.md` (§3.7)
- `analysis/novelty_coverage.py` → `RESULTS_NOVELTY.json` (§3.8)
- `analysis/chaos_diagnostics.py` → `RESULTS_CHAOS.json` (§3.9)
