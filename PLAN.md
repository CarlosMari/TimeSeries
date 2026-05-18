# PLAN.md — Path to submission (post-pivot)

**Today:** 2026-05-18 (post-pivot mid-execution)  **Target:** submitted by end of August 2026 (slipped from end of July; quality > deadline per user direction).

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
| A | Data pipeline rebuild (no-sort) + retrain models 1, 2, 3 × 3 seeds | seed-42 row ✓ 2026-05-15→16; seeds 123 + 2026 queued |
| B | Models 4, 5, 6, 7 implemented + trained × 3 seeds | seed-42 row ✓; seed-123 row: m4 ✓, m5 ✓, m7 queued, **m6 killed mid-train at 44% to free GPU for B1 (will resume after B1)**; seed-2026 not started; B1 sub-batch in flight |
| C | Unified eval harness + RESULTS_COMPARATIVE.json + OOD test sets | harness + D3 fix + OOD sets ✓; **seed-42 row of RESULTS_COMPARATIVE.json populated 00:15 UTC 2026-05-17** (all 7 models); seed-123/2026 in progress |
| D | Synthetic-data lens-validation experiment | **DONE** 2026-05-15 |
| E | Comparative figures | seed-42 regen ✓ 00:15 UTC 2026-05-17 (`fig_comparative_table`, `fig_recon_vs_chaos`); multi-seed regen after seeds 123 + 2026 land |
| F | First full draft (methods + results + discussion) | pending |
| G | Supplement + reviewer pre-mortem + advisor review | pending |
| H | Final polish + submission | pending |

---

## Phase A — Data pipeline rebuild + first 3 models

**Goal:** TRAIN_FINAL_NOSORT.pkl + TEST_FINAL_NOSORT.pkl on disk; models 1, 2, 3 trained with 3 seeds each (9 checkpoints).

- [x] **A1** Added `--no-sort` flag to `data_generation/preprocessor.py`. Both `TRAIN_FINAL_NOSORT.pkl` (117k) and `TEST_FINAL_NOSORT.pkl` (39k) on disk; round-trip identity confirmed at machine epsilon.
- [x] **A2** Wrote `train_pivot.py` thin CLI wrapper (preserves v1 `python train_cvae.py` reproducibility bit-identically). Accepts `--model`, `--seed`, `--data-train`, `--epochs`, `--use/no-scale-conditioning`, `--wandb-project`. Fixed TF-schedule divide-by-zero at small epoch counts.
- [x] **A3** Model 1 seed 42 ✓ (`model_1_seed42.pth`, 13:19 UTC 2026-05-15). Seeds 123 + 2026 queued via `scripts/autoqueue.sh`.
- [x] **A4** Model 2 (no-cond) seed 42 ✓ (`model_2_seed42.pth`, 17:10 UTC 2026-05-15). Seeds 123 + 2026 queued.
- [x] **A5** Model 3 (`StochasticLSTMVAE` at `src/models/cvae_stochastic.py`) — learnable σ via softplus, noise injected on the decoder hidden state at every step. Smoke-tested. Queued × 3 seeds.
- [x] **A6** Unified eval harness `analysis/evaluate_all_models.py` validates on the v1 checkpoint (reproduces R² 0.93 norm, 0.97 orig, max-val R² 0.97).

**Phase A status:** infrastructure done; training in progress under `scripts/autoqueue.sh`.

---

## Phase B — Models 4–7

- [x] **B1** Model 4 — `src/models/latent_ode.py` (Rubanova 2019: ODE-RNN encoder → posterior on z₀, `f_θ(z,t)` MLP, torchdiffeq.odeint, MLP decoder). Scale conditioning preserved. Smoke-tested. **Queued for training × 3 seeds.**
- [x] **B2** Model 5 — `src/models/transformer_vae.py` (4-layer Transformer encoder/decoder, positional encoding, learned query embedding, cross-attention from z). Smoke-tested. **Queued × 3 seeds.**
- [x] **B3** Model 6 — `src/models/kan_vae.py` (LSTM backbone + KAN heads via vendored `efficient_kan`). Avoided pip-install of efficient-kan because its `torch>=1.5` constraint silently upgraded torch to cu130 — incompatible with the system CUDA 12.7 driver. Vendored the source file directly. Smoke-tested. **Queued × 3 seeds.**
- [x] **B4** Model 7 — `src/models/glv_regression.py` + `train_glv_regression.py` (LSTM → MLP → (r̂, Â); MSE on `PARAM_RECOVERY_MATCHED.pkl` ground truth; inference via solve_ivp). **Queued × 3 seeds.**

**Phase B status (refreshed 2026-05-18 09:21 UTC):** seed-42 row complete for all 7 models. Seed-123 row: m1, m2, m3, m4, m5 ✓; m6 (KAN-VAE) was 44% in when killed at 06:10 to free GPU for B1 (PROJECT.md §1.2.1 watcher postmortem); m7 queued. Seed-2026 batch hasn't started.

**Phase B1 sub-batch (interleaved):** 4 trainings (m3 frozen-σ at 0.05/0.10/0.20 + m1 spectral-loss 0.1) running now. First variant (frozen-σ 0.05) at 90% as of 09:21; projected B1 complete ~20:30 UTC tonight. Autoqueue SIGSTOPped during B1; will resume m6_seed123 → m7_seed123 → seed-2026 batch automatically.

Outstanding from m6 seed-123 kill: KAN seed-123 ckpt will need to be retrained (autoqueue does this automatically on resume) or skipped if compute tight (design doc pre-authorized).

---

## Phase C — Unified eval harness + OOD + comparative results

- [x] **C1** `analysis/evaluate_all_models.py` — per-model adapters for all 7 architectures, dispatches uniform recon + Lens-1 + Lens-2 + Lens-3 eval. Validated on the v1 checkpoint: reproduces R² 0.93/0.97/0.97 and RQA/Lyapunov KS p-values exactly. Writes incrementally so a crash never loses work.
- [x] **C2** D3 fix baked into the eval harness (`DROPPED_FEATURES = {"mean_extrema", "mean_curv"}`). Re-verified on the v1 model: **MMD p = 0.005 (was 0.002)** — the headline finding survives the feature-set cleanup. **Coverage/density jumped from 0.13/0.25 to 0.95/0.98** — the model covers the real distribution well at coarse scales; the mismatch the protocol detects is specifically in fine-grained NLD invariants. Written into PROJECT.md §3.8.1.
- [x] **C3** OOD test sets generated: `data/TEST_OOD_Exp1.pkl` and `data/TEST_OOD_Exp5.pkl` (5k samples each, raw + no-sort-preprocessed + ground-truth `(r, A)` stored). Evaluation pass will run after stage-1 training finishes (waiting on m6 to complete the seed-42 row).
- [ ] **C4** Statistical-comparison plan (Wilcoxon signed-rank + FDR + bootstrap CIs) — not yet implemented; will go into the final eval pass.

**Phase C status:** infrastructure complete; awaiting trained checkpoints to populate the comparative JSON.

---

## Phase D — Synthetic-data lens-validation — DONE ✓ (2026-05-15)

- [x] **D1** `analysis/lens_validation.py`: 5 perturbations × 4-5 metrics, n = 200, results in `RESULTS_LENS_VALIDATION.json`.
- [x] **D2** Figure: `final figures/fig_lens_validation.{pdf,png}` — recon-R² bar plot + lens p-value heatmap.

**Findings (validated paper Section 5):**
- (a) **low-pass filter**: recon-R² = 0.99 (looks fine!) BUT RQA-DET KS p < 10⁻⁸¹, RQA-L_mean p < 10⁻³⁵ → **Lenses 2/3 catch what recon misses.**
- (b) **HF noise**: recon-R² = 0.99 BUT all three lenses at p < 10⁻²² → **Lenses 1/2/3 catch.**
- (c) **amplitude rescale**: recon-R² = 0.44; **only Lens 1 detects** (as designed — Lenses 2/3 are normalization-invariant by construction).
- (d) **phase shift**: recon-R² = 0.84; **no lens detects.** Honest limitation.
- (e) **species permutation**: recon-R² = -0.60; **no lens detects** (protocol is permutation-invariant by construction).

The lenses are sensitive exactly where they should be (low-pass, HF noise — the failure mode of regress-to-mean decoders) and insensitive exactly where they shouldn't be (amplitude, permutation — both are protocol design choices). The phase-shift insensitivity is a genuine limitation to disclose. This is a strong methodological foundation for the paper.

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
