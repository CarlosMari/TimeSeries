# PLAN.md — Path to submission (post-pivot)

**Today:** 2026-05-15  **Target:** submitted by end of August 2026 (slipped from end of July; quality > deadline per user direction).

PLAN.md is the **todo list**; PROJECT.md is the **wiki**. The canonical design is `docs/superpowers/specs/2026-05-15-comparative-evaluation-design.md`. Updated state and findings go into PROJECT.md first; this file gets pruned to reflect what remains.

---

## Pivot in one line

The paper became a **comparative empirical study of 7 generative architectures on GLV trajectories under a 3-lens nonlinear-dynamics evaluation protocol**, not a single-model paper about a scale-conditioned VAE.

---

## Guiding principles (post-pivot)

1. **Quality > deadline.** User has explicit permission for end-July → end-August slip.
2. **Three seeds per model.** Single-seed numbers are not the final paper's numbers.
3. **Same eval matrix for all 7 models.** No model gets a custom metric.
4. **Honest negatives are findings.** If model X fails to converge in 2 days, document and drop; if model Y wins on RQA but loses on recon, that's an *interesting* result.
5. **Update PROJECT.md and REFERENCES.md as we go.** Don't let stale framing accumulate.
6. **Run autonomously.** Per user direction, don't ask for permission on individual training runs / refactors. Confirm only on destructive ops.

---

## Phase tracker

| Phase | What | Status |
|---|---|---|
| 0 | (legacy) Lock down v1 numbers + bug fixes + RESULTS.json | **DONE** 2026-05-14 |
| 0.5 | (legacy) Strengthen v1 science (recoverability, baseline, novelty) | **DONE** 2026-05-14 |
| 0.7 | (legacy) Chaos diagnostics on v1 | **DONE** 2026-05-15 |
| **PIVOT** | **Decide + write design doc + seed REFERENCES.md** | **DONE** 2026-05-15 |
| A | Data pipeline rebuild (no-sort) + retrain models 1, 2, 3 × 3 seeds | **NEXT** |
| B | Models 4, 5, 6, 7 implemented + trained × 3 seeds | pending |
| C | Unified eval harness + RESULTS_COMPARATIVE.json + OOD test sets | pending |
| D | Synthetic-data lens-validation experiment | pending |
| E | Comparative figures (replaces most of `final figures/`) | pending |
| F | First full draft (methods + results + discussion) | pending |
| G | Supplement + reviewer pre-mortem + advisor review | pending |
| H | Final polish + submission | pending |

---

## Phase A — Data pipeline rebuild + first 3 models

**Goal:** TRAIN_FINAL_NOSORT.pkl + TEST_FINAL_NOSORT.pkl on disk; models 1, 2, 3 trained with 3 seeds each (9 checkpoints).

- [ ] **A1** Add a `--no-sort` flag (or new function) in `data_generation/preprocessor.py` that skips step 2 (sort-by-peak). Run on existing raw training & test seeds. Produce `TRAIN_FINAL_NOSORT.pkl`, `TEST_FINAL_NOSORT.pkl`. Sanity-check: shapes match v1 (117k / 39k); denormalization round-trip identity check.
- [ ] **A2** Patch `train_cvae.py` to accept a data-path argument and a seed argument. Confirm WandB project name change to `Conditional_LV_VAE_pivot` so we don't clutter the v1 history.
- [ ] **A3** Train **model 1** (scale-conditioned LSTM-VAE) on no-sort, seeds {42, 123, 2026}. Save to `model_ckpts/model_1_seed{42,123,2026}.pth`. ~3 hr each.
- [ ] **A4** Train **model 2** (no-conditioning) on no-sort, seeds {42, 123, 2026}. ~3 hr each. Same hyperparameters as model 1 except `use_scale_conditioning=False`.
- [ ] **A5** Implement **model 3** (stochastic-decoder): modify `src/models/cvae.py` decoder LSTM to inject Gaussian noise in hidden state with learned σ. Add small entropy bonus to prevent σ collapse. Train × 3 seeds.
- [ ] **A6** Run v1 eval harness adapted (recon R², max-val R², latent dims) on each of the 9 new checkpoints. Spot-check: model 1 numbers should be close to v1 (~R² 0.95 on no-sort, possibly different by 0.01-0.02). If radically different, debug before proceeding.

**Phase A exit criterion:** 9 checkpoints + 9-row provisional results table for models 1–3.

---

## Phase B — Models 4–7

- [ ] **B1** Model 4 — Latent-ODE. New file `src/models/latent_ode.py`. Encoder: existing LSTM. Latent dynamics: 2-layer MLP `f_θ(z, t)`. Integrator: `torchdiffeq.odeint`. Decoder: same as model 1 (autoregressive on z(t)). Train × 3 seeds. ~5 hr each.
- [ ] **B2** Model 5 — Transformer-VAE. New file `src/models/transformer_vae.py`. 4-layer encoder + 4-layer decoder with positional embeddings. CLS pooling for posterior. Train × 3 seeds. ~4 hr each.
- [ ] **B3** Model 6 — KAN-VAE. New file `src/models/kan_vae.py`. Pip-install `efficient-kan`. Replace MLP heads with KAN layers; LSTM backbone unchanged. Train × 3 seeds. Time-box at 2 days per seed; fallback to KAN-output-head-only if unstable.
- [ ] **B4** Model 7 — Direct GLV regression. New file `src/models/glv_regression.py`. LSTM encoder → MLP → (r̂, Â). Train on `PARAM_RECOVERY_MATCHED.pkl` (10k). Inference: `solve_ivp` with predicted (r̂, Â). Train × 3 seeds. ~1 hr each.

**Phase B exit criterion:** 12 additional checkpoints. All 21 checkpoints saved.

---

## Phase C — Unified eval harness + OOD + comparative results

- [ ] **C1** `analysis/evaluate_all_models.py`: loops over all 21 checkpoints, dispatches to per-model adapters (because each architecture's inference call signature differs slightly), runs the full eval matrix, writes per-model JSON + the aggregate `RESULTS_COMPARATIVE.json`.
- [ ] **C2** Drop `mean_extrema` + `mean_curvature` from the feature-MMD vector (D3 fix). Re-verify the §3.8 finding (real vs gen distinguishable) on v1, document if KS p-value of the residual feature set changes the headline.
- [ ] **C3** Generate OOD test sets: `data/TEST_OOD_Exp1.pkl` (5k trajectories, `r ~ Exp(1)`), `data/TEST_OOD_Exp5.pkl` (5k, `r ~ Exp(5)`). Preprocess without sort. Evaluate all 21 checkpoints on both.
- [ ] **C4** Statistical-comparison plan applied to RESULTS_COMPARATIVE: Wilcoxon signed-rank with FDR correction on continuous metrics; KS ladder for distributional metrics; bootstrap CIs.

**Phase C exit criterion:** RESULTS_COMPARATIVE.json populated; one Pandas-printable 7-row × N-column table that is the paper's central result.

---

## Phase D — Synthetic-data lens-validation

- [ ] **D1** `analysis/lens_validation.py`: take 200 real GLV trajectories; apply (a) low-pass filter, (b) high-frequency noise, (c) amplitude rescale, (d) phase shift, (e) species permutation. For each: compute recon-R² vs original *and* 3-lens distance. Tabulate sensitivity.
- [ ] **D2** Figure: 5 perturbations × 4 metrics (recon-R², MMD, RQA-DET, λ₁) showing the lens protocol detects (a, b, d) while recon-R² remains high.

This is the methodological lynchpin. Without it the eval-protocol contribution is weak.

---

## Phase E — Comparative figures

- [ ] Headline table figure (7 models × all metrics).
- [ ] RQA-ladder real vs each-model on (DET, L_mean, L_max, LAM, TT).
- [ ] Lyapunov-distribution panel (7 models overlaid on real).
- [ ] Parameter-recoverability panel (model 7 should win; quantify by how much).
- [ ] Latent-ODE vs LSTM-VAE chaos-comparison panel (the prediction-vs-test panel).
- [ ] Stochastic-decoder vs deterministic-decoder ablation panel (the causal test).
- [ ] Lens-validation figure (from Phase D).

---

## Phase F — Draft

- [ ] **F1** Methods: data pipeline, 7 model architectures, training schedule, eval-protocol formalization.
- [ ] **F2** Results: read every number from `RESULTS_COMPARATIVE.json`. Subsections: reconstruction comparison, distributional fidelity, NLD invariants, parameter recoverability, OOD generalization, lens validation, stochastic-decoder causal test.
- [ ] **F3** Discussion: what the 7-way comparison tells us about inductive biases; the recurrence-prior vs. ODE-prior vs. attention-prior story; honest limitations (single dataset family, T=65, etc.).
- [ ] **F4** Introduction (write last): why generative models for ecological dynamics matter, why standard eval metrics miss dynamical defects, what our protocol contributes, why this venue.
- [ ] **F5** Abstract (write last).

---

## Phase G — Supplement / pre-mortem / review

- [ ] Full per-seed tables (instead of just mean ± std) in supplement.
- [ ] Per-model failure-mode analyses.
- [ ] Hyperparameter-sweep tables.
- [ ] "Anticipated objections" doc; address each in the paper or in a response-to-reviewer plan.
- [ ] Advisor + co-author review.

---

## Phase H — Submit

- [ ] Address advisor feedback.
- [ ] elsarticle template, vector figures, BibTeX, cover letter, suggested reviewers.
- [ ] Submit.

---

## Risk register (post-pivot)

| Risk | Mitigation |
|---|---|
| 21 training runs is too much compute | Sequential queueing; each model checkpointable. Worst-case fallback: 1 seed per model (still better than the v1 single-seed numbers). |
| Some new architecture (Transformer / KAN) doesn't converge | 2-day time-box per model. Document failure honestly and drop. |
| Stochastic-decoder ablation gives null result | Useful finding — "decoder stochasticity is insufficient" is a legitimate paper claim. |
| Latent-ODE wins decisively on chaos but loses on recon | Best outcome for the paper: clean architectural trade-off story. |
| Sort-step removal hurts the existing v1 R² materially | Accept honest worse numbers if correct. The paper's headline is comparative, not absolute. |
| OOD experiment shows all models fail | "Scope is `r~Exp(2)`, 7 species" becomes the honest scope claim. Still publishable. |
| KAN-VAE training is the bottleneck (notoriously fiddly) | Time-box hard. Skip if no convergence in 2 days. |
| The 3-lens protocol turns out to be insensitive on the new models | Lens-validation experiment (Phase D) catches this early before we commit to the framing. If lens validation shows the lenses are weak, regroup. |

---

## Anti-goals (post-pivot)

- Train an 8th model unless we drop one first.
- Add a new dataset (different ODE system) before submitting.
- Add new evaluation metrics beyond the locked eval matrix.
- Refactor the codebase beyond what each model's implementation requires.

---

## Future-work bucket (deferred — v2 / follow-up paper)

- Physics-informed losses (GLV residual penalty).
- Apply the 3-lens protocol to a *different* dynamical-system family (Rössler, double pendulum, etc.) as a generality demonstration — natural follow-up paper.
- Long-horizon extrapolation (T=65 → T=200).
- GP emulator as an 8th comparator (revision-cycle add if reviewer asks).
- Active learning for under-represented regimes.
- Hierarchical CVAE with explicit interaction-matrix prior.

---

## Idea log (raised during the pivot brainstorm — not promised, will revisit during write-up)

- **Per-regime chaos diagnostics.** Bin samples by real-data λ₁ quartile or dominant frequency; report each model's NLD-preservation by regime. ~10 min of code added to `chaos_diagnostics.py`. Strong supplement candidate.
- **Decoder-noise scaling sweep.** Instead of one stochastic-decoder model, sweep σ ∈ {0, 0.01, 0.05, 0.1, 0.2}. Trace λ₁ as a function of σ. Defer to revision unless cheap.
- **Mode-coverage decomposition.** Decompose the smoother-than-real defect into amplitude vs. phase vs. frequency components. Could yield a four-axis "fidelity radar" for each architecture. Aspirational; defer to v2.
