# Design Spec — Comparative Evaluation of Generative Models of GLV Dynamics under a 3-Lens Nonlinear-Dynamics Protocol

**Date:** 2026-05-15
**Status:** Approved (verbal; this doc supersedes the prior single-model paper plan)
**Owners:** Carlos Mari Armengol (lead); two physics-professor advisors
**Target venue:** *Chaos, Solitons & Fractals* (Elsevier), or comparable nonlinear-dynamics venue
**Supersedes:** the single-model paper framing in PROJECT.md / PLAN.md as of `f5a953e`

---

## 1. What changed and why

We have spent ~18 months building a scale-conditioned LSTM-VAE for Generalized Lotka–Volterra trajectories. The model works (recon R² = 0.97, max-value R² = 0.97 vs. baseline 0.59), but three independent diagnostics — feature-MMD, RQA, and largest Lyapunov exponent — show that **generated trajectories are quantifiably less chaotic than real ones** (DET 0.99 vs 0.59; λ₁ +0.03 vs +0.08; KS p-values 10⁻⁴³ to 10⁻⁹⁸).

A pre-Phase-4 audit identified that the current paper's framing has three structural problems:

- The contribution ("a scale-conditioned VAE works") is incremental and small in scope.
- Three independent metrics show a real generative defect, framed as a "limitation" — reviewers will read it as "the central object doesn't fully work."
- No comparison against a real comparator (latent-ODE, neural-ODE, GP, etc.); only against an internal ablation.

These observations argue against an incremental write-up and for a structural pivot. We adopt a hybrid #1+#3 framing from the brainstorm:

> **A nonlinear-dynamics evaluation protocol for generative models of dynamical systems, applied to seven architectures on Generalized Lotka–Volterra trajectories.**

The paper's contribution becomes (a) the protocol itself, (b) the comparative empirical study that demonstrates it discriminates. The VAE is one of seven worked examples; its limitations become demonstrative evidence that the protocol works rather than embarrassments.

The contribution arc is:

1. **A protocol** (feature-MMD + RQA + Lyapunov + KS) for evaluating generative-model fidelity in nonlinear-dynamics space, more sensitive than reconstruction-R² alone.
2. **A comparative study** of 7 generative architectures on a controlled testbed (GLV), measured under that protocol.
3. **A causal demonstration** that the protocol's findings are correct: the protocol identifies "deterministic decoder → mode-covering" as a likely cause of the smoother-than-real defect in LSTM-VAEs; we test this directly by including a stochastic-decoder variant.
4. **A clean scientific finding** about which inductive biases preserve which dynamical invariants.

---

## 2. Decisions locked

| Decision | Choice |
|---|---|
| Paper identity | Hybrid eval-method + comparative-study (option #1 + #3) |
| Number of models | 7 (lineup §3) |
| Seeds per model | 3 (proper CIs, no single-seed asterisks) |
| Sort step in preprocessing | **Drop** — retrain canonical VAE without it (fix D1) |
| Novelty feature set | Drop `mean_extrema` + `mean_curvature`, re-verify §3.8 finding (fix D3) |
| Out-of-distribution test | **Yes** — held-out family with `r~Exp(1)` and `r~Exp(5)` (fix F4) |
| Hyperparameter protocol | Uniform 3-point sweep on held-out val set (fix F1) |
| Deadline | Quality > deadline; expect end-July to slip to end-August |

---

## 3. The 7-model lineup

All models train on **the same data** (TRAIN_FINAL_PROCESSED.pkl with the sort step removed; we will generate a new processed file: `TRAIN_FINAL_NOSORT.pkl`). All models are evaluated under **the same eval matrix** (§4).

| # | Model | Inductive bias | New? | Cost (1 seed) |
|---|---|---|---|---|
| 1 | **Scale-conditioned LSTM-VAE** | Recurrent, scale-aware autoregressive deterministic decoder | retrain w/o sort | ~3 hr |
| 2 | **No-conditioning LSTM-VAE** | Recurrent, no scale info — ablation | retrain w/o sort | ~3 hr |
| 3 | **Stochastic-decoder LSTM-VAE** | Recurrent + decoder Gaussian noise (tests deterministic-decoder hypothesis) | new | ~3 hr |
| 4 | **Latent-ODE** | Continuous-time ODE prior in latent space (Rubanova et al. 2019) | new | ~5 hr |
| 5 | **Transformer-VAE** | Attention; no recurrence inductive bias | new | ~4 hr |
| 6 | **KAN-VAE** | Kolmogorov–Arnold function-approximation basis instead of MLP | new | ~5 hr |
| 7 | **Direct GLV regression** | MLP → (r, A); generate by numerical integration (physics-naive parameter recovery) | new | ~1 hr |

Total: 7 × 3 = 21 training runs. Roughly 80–120 GPU-hr cumulative; ~3 weeks of wall-clock time on one GPU if queued. We will use the existing `train_cvae.py` for VAE-family models with appropriate module swaps; latent-ODE / Transformer / KAN / GLV-regression are new training scripts.

### Why these seven (one-line each)

- **1, 2** isolate the scale-conditioning contribution (the smaller paper's original headline; now part of a broader story).
- **3** tests the *specific causal hypothesis* the protocol surfaces from §3.8 / §3.9 of the current PROJECT.md.
- **4** is the obvious "physics-aware" comparator a CSF reviewer expects. Continuous-time ODE prior should preserve dynamical invariants better than an autoregressive RNN — *if* the protocol's claims are right, this should be visible in RQA/λ₁.
- **5** isolates the recurrence inductive bias. Transformers process the whole sequence at once; if the smoother-than-real defect is recurrence-specific, Transformer-VAE should look different.
- **6** is novel-architecture coverage. KAN is recent (2024), the literature is evaluating it on standard tasks, and ours is one of the first to use it as a VAE backbone for time-series. If it works, that's a strong by-product result.
- **7** is the physics-informed inverse-problem baseline. Predict GLV parameters, then integrate. Should win on parameter recovery and lose on data-driven flexibility — testing that prediction with the protocol is one of the paper's planned findings.

---

## 4. Evaluation matrix (applied uniformly to all 7)

| Group | Metric | Notes |
|---|---|---|
| Reconstruction | R² (normalized + original-scale), MAE, MSE per curve | Existing infra |
| Generation quality | Nearest-neighbor distance to test set; memorization ratio | Existing infra |
| Scale prediction | Max-value R² per curve and pooled | Existing infra; N/A for model 7 |
| Distributional fidelity (Lens 1) | Feature-MMD (Gaussian kernel) + permutation p-value; density/coverage@5 | Adapted from `analysis/novelty_coverage.py`. Feature set excludes `mean_extrema` + `mean_curvature` (D3 fix). |
| NLD invariants (Lens 2 + Lens 3) | RQA: DET, L_mean, L_max, LAM, TT (Lens 2); Rosenstein λ₁ (Lens 3); KS-tested real vs. each model | Existing infra in `analysis/chaos_diagnostics.py` |
| Latent structure | Active dims, PCA variance curve, dim↔feature correlation matrix | N/A for models 4 (latent-ODE has a different latent), 7 (no latent) — report where applicable |
| Parameter recoverability | Ridge `μ(z) → (r, A)` on a fresh matched set | Existing infra; model 7 trivially recovers (r, A); compare others |
| Out-of-distribution (F4) | All above, on `r~Exp(1)` and `r~Exp(5)` held-out family sets | New: generate 5k trajectories each, ~30 min |
| Compute / size | Training time, parameter count, inference time | Tabulate |

All metrics are reported as **mean ± std across 3 seeds** with bootstrap or two-sample CIs where appropriate.

### Statistical comparison plan

The headline empirical table is a 7-row × N-metric grid. To make claims of the form "model X is better than model Y on metric M" we will use:

- For continuous metrics (R²): paired comparisons across the 3 × 39k test samples; Wilcoxon signed-rank with FDR correction across the metric table.
- For distributional metrics (RQA, λ₁): KS test of the per-sample distribution real vs. each model; report the KS-statistic ladder.
- For binary outcomes (% > 0.9, etc.): bootstrap CIs.

---

## 5. The synthetic-data sanity check (Lens-validation)

To support the claim "the 3-lens protocol detects defects that recon-R² misses," we add **one synthetic experiment** to Methods:

- Take a small set of real GLV trajectories.
- Apply controlled perturbations: (a) low-pass filter (smoothing); (b) high-frequency noise injection; (c) amplitude rescaling; (d) phase shift; (e) species permutation.
- For each perturbation, measure the recon-R² between perturbed and original *and* the 3-lens distance.
- Show that for perturbations (a, b, d) recon-R² remains high while the 3-lens protocol detects the change — i.e., the lenses are sensitive to dynamically-relevant defects.

This is a ~1-week experiment with simple infrastructure. It is the lynchpin of the "method is novel" argument.

---

## 6. Architecture sketches for new comparators

Detail level: enough to start implementation, not enough to remove all design choices. Final hyperparameters fixed by the §F1 protocol after a small sweep.

### Model 3 — Stochastic-decoder LSTM-VAE
Identical to model 1; modify decoder LSTM to add `+ σ · ε_t` with `ε_t ~ N(0, I)` to the hidden state at each timestep. σ is a learned scalar (initialized 0.05) or a small per-dim vector. Reparameterization trick keeps gradients flowing. KL term against the prior is unchanged; we add a small entropy bonus on the decoder samples if needed to prevent σ → 0.

### Model 4 — Latent-ODE
Encoder: bidirectional LSTM over (7, 65) → posterior `(μ_0, σ_0) ∈ ℝ^{30}` for the *initial latent state*. Latent dynamics: `dz/dt = f_θ(z, t)` with `f_θ` a small MLP (2 layers, hidden 128, tanh). Integrate `z(t)` over the 65 timesteps with `torchdiffeq.odeint`. Decoder: same as model 1 (autoregressive LSTM on z(t) → trajectories) — *or* a simple MLP per-timestep. Try both, pick the lower-validation-loss variant.

### Model 5 — Transformer-VAE
Encoder: 4-layer Transformer encoder over (7, 65) with learned positional embeddings → pooled CLS-style representation → posterior `(μ, log σ²) ∈ ℝ^{30}`. Decoder: 4-layer Transformer decoder, masked self-attention, cross-attention over a fixed-length "memory" projected from z. Teacher forcing as in model 1.

### Model 6 — KAN-VAE
Same overall structure as model 1, but replace MLP layers (in encoder projection, decoder output head, max-value head) with KAN layers from the `efficient-kan` PyTorch implementation. LSTM backbone is unchanged (KAN doesn't replace recurrence; we are testing whether KAN as a function approximator changes the result downstream of recurrence). If KAN proves unstable, fallback is "KAN at the output head only."

### Model 7 — Direct GLV regression
Encoder: same as model 1's LSTM-encoder (bidir, hidden 256). Output: `(r̂, Â) ∈ ℝ^{7+49}` (no decoder, no KL, no latent). Loss: MSE on `(r, A)` on the matched dataset where ground-truth is available (10k samples). At inference time: predict `(r̂, Â)`, then `solve_ivp` to generate the trajectory. Note that this model requires the matched dataset for training, so it only sees 10k samples vs. 117k for the others — flag this explicitly in the comparison; consider generating 100k matched samples if needed (~1 day of CPU integration).

---

## 7. Schedule (best-effort; quality > deadline)

| Week | Block | Output |
|---|---|---|
| 1 (now) | Design doc + PROJECT.md/PLAN.md/README rewrite; data pipeline rebuild without sort | Doc, `TRAIN_FINAL_NOSORT.pkl`, retrain script |
| 2 | Models 1, 2, 3 trained × 3 seeds; D3 feature-set fix | 9 checkpoints, updated novelty results |
| 3 | Models 4, 7 trained × 3 seeds; synthetic-data sanity check | 6 checkpoints, lens-validation figure |
| 4 | Models 5, 6 trained × 3 seeds; first uniform eval pass | 6 checkpoints, RESULTS_COMPARATIVE.json (v1) |
| 5 | F4 OOD experiment on all 7 models; final RESULTS_COMPARATIVE.json | OOD numbers, full comparative table |
| 6 | All paper figures regenerated under new framing | Final figure inventory |
| 7–8 | First full draft of paper (methods, results, discussion) | `paper.tex` v1 |
| 9 | Supplement; advisor review | Supplement |
| 10–11 | Revisions; final polish | `paper.tex` final |
| 12 | Submission | — |

Earliest submission: ~end of August 2026. We accept this slip.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 21 training runs is too much compute | Queue sequentially; each model is independently checkpointable. Worst-case fallback: 1 seed each (still better than current paper). |
| Some comparator just doesn't train (Transformer-VAE / KAN-VAE notoriously fiddly) | Time-box at 2 working days per model. If non-converging after 2 days, drop it and document in Methods as "we attempted X with [config]; it did not converge in our budget." Honest exclusion is fine. |
| Stochastic-decoder ablation gives null result | Still informative — paper claims "decoder stochasticity is insufficient, suggesting the defect is deeper than mode-covering." |
| Latent-ODE wins everything decisively | Best outcome — clean scientific finding, paper has a positive result. |
| Latent-ODE loses to LSTM-VAE | *Also* informative — the continuous-time prior is not the panacea reviewers might assume. |
| Reviewer wants GP emulator we excluded | Acceptable revision-cycle add. We make the decision-tree explicit in the Methods section. |
| Sort-step removal hurts the existing R² numbers materially | Possible. We accept honest worse numbers if they are correct. The framing of the paper is not "high R²" but "comparative". |
| OOD held-out family experiment kills the paper (all models fail catastrophically) | Frame as "scope limitation: trained-and-evaluated on r~Exp(2), 7 species; held-out family failures characterize the regime of applicability." Honest scope claim. |

---

## 9. What gets thrown away

- The §3.7 "baseline comparison: conditioned vs non-conditioned" headline framing. Folded into the broader §3 of the new paper. **Numbers will be re-derived on the no-sort pipeline; expect them to shift (likely smaller margin on max-value R² because the no-sort task is harder, but the qualitative finding "scale conditioning helps" should hold; if it doesn't, that itself is a paper finding).**
- The §4.5 "honest framing: model identifies dominant species, blind to coupling" narrative. Becomes one model's row in the parameter-recovery comparison. **With the sort removed, the off-diagonal R² may improve materially because the model is no longer recovering permuted entries; the finding may change from "blind to coupling" to "partially recovers coupling," which would be a stronger result.**
- All current model checkpoints continue to exist (no deletion) but they are no longer the headline. The retrained-without-sort versions are.
- The "final figures/" directory is mostly archival once the new comparative figures are produced. Keep for the appendix.

## 10. What is preserved as-is

- The 3-lens protocol code (`analysis/novelty_coverage.py`, `analysis/chaos_diagnostics.py`). Both are reused, with the D3 feature-set fix applied to the first.
- The data generation pipeline (`data_generation/generate_family_FIXED.py` etc.). The preprocessing pipeline is rebuilt without the sort step.
- The bootstrap-CI metrics infrastructure in `analysis/produce_paper_metrics.py`. Generalized to handle multiple models.
- The matched-dataset for parameter recoverability. Reused as the training set for model 7 (or regenerated at larger size if needed).

---

## 11. Author/reader transparency

We will state in the paper that this is a **comparative empirical study**, not a benchmark — we are deliberately comparing models on a single dataset (GLV) to characterize dynamical-preservation properties, not to crown a winner across dataset families. CSF readership is comfortable with case-study framing; we will not overstate generality.

---

## 12. Next steps after this design doc

1. User reviews this doc, requests changes inline or signs off.
2. Update PROJECT.md (rewrite §§ 1, 2.3, 7, 8 to reflect pivot; mark old framing as superseded).
3. Update PLAN.md (replace phase structure with the §7 schedule above).
4. Update root README.md to reflect new paper identity.
5. Invoke writing-plans skill to produce an implementation plan from this design doc (the data-pipeline rebuild without sort is the first concrete coding task).
6. Commit: design doc + updated wiki.
