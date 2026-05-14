# PLAN.md — Path to submission

**Today:** 2026-05-14  **Hard target:** **end of July 2026** (submitted, ideally under review).  **Soft target:** mid-July.

This project has been running too long. The goal is to *finish it*, not perfect it. New ideas get weighed against "does this ship the paper or delay it." Default to ship.

PLAN.md is the **todo list**; PROJECT.md is the **wiki**. Anything new (data, results, decisions, course corrections) goes into PROJECT.md first, then this file gets pruned to reflect what remains.

---

## Guiding principles

1. **No new model architecture.** The 30D scale-conditioned CVAE is the model.
2. **Fix things that are wrong before adding things that are missing.**
3. **Reviewer-driven scope.** Three predictable objections from a *Chaos, Solitons & Fractals* (CSF) reviewer:
   - "Compare to a baseline."
   - "What about chaos-specific diagnostics (Lyapunov, RQA)?"
   - "Is the model just memorizing the training distribution?"
4. **Time-box.** If something hasn't yielded a usable result in 2 working days, cut it.

---

## What the paper's headline result is now

Phase 2 results in hand, the paper's structure is:

1. **Main quantitative contribution (now isolated against baseline):** scale-conditioned VAE for GLV achieves max-value $R^2 = 0.97$ vs non-conditioned baseline at $0.59$ (negative per-curve on 5/6 targets). The architectural change is decisively responsible.
2. **Generative validation:** generated samples obey LV equations at mean $R^2 = 0.97$ (real data is at $R^2 = 0.99$); gap is 0.013 — within the noise band.
3. **Honest interpretability story (the new headline):** $\mu(z)$ partially recovers GLV parameters — dominant-species growth rates at $R^2 = 0.43$, diagonal of $A$ at $R^2 = 0.20$, cross-species coupling at $R^2 = 0.03$. **The model identifies parameters of the species it can see, and is blind to the rest.** Consistent with species-centric latent geometry and with §3.8.
4. **Honest limitation (also from §3.8):** generated trajectories are *smoother / less oscillatory* than real (MMD permutation p = 0.002, `mean_extrema` KS = 0.98). Classic VAE mode-covering. Concrete future-work direction.

That story is publishable at CSF. Every claim is backed by a number with a CI.

---

## Phase tracker

| Phase | What | Status |
|---|---|---|
| 1 | Lock down numbers (bug fixes + RESULTS.json + figure cleanup) | **DONE** 2026-05-14 |
| 2 | Strengthen the science (recoverability, baseline, novelty) | **DONE** 2026-05-14 |
| 3 | Chaos diagnostics (RQA + Lyapunov) | **NEXT** |
| 4 | First full draft (methods + results) | not started |
| 5 | Refine + supplement + reviewer pre-mortem | not started |
| 6 | Co-author / advisor review | not started |
| 7 | Final polish + submission | not started |

---

## Phase 1 — DONE (recap)

- [x] LV-validation bug fixed (`sample_norm[0]` species-axis collapse). Real LV-$R^2$ = 0.99 median; gen with θ=0.005 = 0.99 median; gap of 0.013 in mean.
- [x] `analysis/produce_paper_metrics.py` = single metrics source of truth → `RESULTS.json` + `RESULTS.md` with bootstrap 95% CIs.
- [x] Stale/wrong docs deleted (8 files), stale duplicate scripts deleted (8+ files).
- [x] β-schedule wart fixed in `train_cvae.py`.
- [x] `.gitignore` tightened (LaTeX intermediates, root-level ad-hoc PNG/PKL; removed bad `utils/` exclusion).
- [x] Committed `5d64e64` "Lock down paper-ready state".

---

## Phase 2 — IN PROGRESS

### 2.1 GLV parameter recoverability — DONE ✓ (2026-05-14)

- [x] Built `analysis/parameter_recoverability.py`: generates fresh 10k matched dataset with seed base 555M, encodes through CVAE, fits Ridge $\mu \to r$ and $\mu \to \mathrm{vec}(A)$ with 5-fold CV, writes JSON + figure.
- [x] Sanity-check: recon $R^2$ on matched data = 0.939, matches the test-set baseline (in-distribution).
- [x] Results: $r$ at $R^2 = 0.24$, $A$ diag at 0.20, $A$ off-diag at 0.03, Re(eig A) at 0.16, Im(eig A) at 0.01. Stored in `RESULTS_PARAM_RECOVERY.json` and `final figures/fig_param_recoverability.pdf`.
- [x] PROJECT.md §4.5 written with the honest framing.

### 2.2 Baseline comparison — DONE ✓ (2026-05-14)

- [x] Trained `model_ckpts/model_final_30_baseline.pth` (1000 epochs, ~7.5 hr).
- [x] `analysis/evaluate_baseline.py` → `RESULTS_BASELINE.json` + `RESULTS_COMPARISON.md`.
- [x] **Headline**: max-value $R^2$ conditioned **0.971** vs baseline **0.591** (Δ +0.381); per-curve, baseline gives negative $R^2$ on 5/6 max-value targets — worse than mean predictor, exactly as the architectural argument predicted.
- [x] Original-scale recon $R^2$: conditioned **0.965** vs baseline **0.844** (Δ +0.121) — baseline's broken max-val prediction destroys the denorm.
- [x] Normalized recon $R^2$: baseline **0.945** vs conditioned **0.933** (Δ −0.011). The expected trade-off: conditioned model spends a fraction of its capacity learning scale.
- [x] PROJECT.md §3.7 written with the comparison table — this is the paper's Table 1.

### 2.3 Novelty / coverage — DONE ✓ (2026-05-14)

- [x] `analysis/novelty_coverage.py` finished — dynamical-feature vectors (26-D), MMD permutation test, density-coverage (Naeem 2020), per-feature KS.
- [x] **Result**: MMD² = 0.068, permutation p = **0.002** → real and generated are statistically distinguishable in feature space. Density@5 = 0.135, coverage@5 = 0.246. 20/26 features differ at p<0.05.
- [x] **Diagnosed the source**: `mean_extrema` KS = 0.983, `mean_curvature` KS = 0.685 — generated trajectories are smoother / less oscillatory than real. Subdominant-species statistics also differ. Species means and dominant frequency match.
- [x] PROJECT.md §3.8 written with the honest framing — limitation reported, concrete future-work direction (decoder stochasticity or perceptual loss for high-frequency content).

**Phase 2 COMPLETE.** All three experiments done; the paper's "results" section now has its full skeleton.

---

## Phase 3 — Chaos diagnostics (after Phase 2)

CSF will expect at least one chaos-specific analysis beyond recurrence plots. Pick **two** of:

- [ ] **RQA (recurrence quantification)** — recommended. For matched real and generated (n=200 each), compute recurrence rate, determinism, mean diagonal line length, laminarity, trapping time. Use `pyts` or `pyRQA`. Compare distributions with KS test.
- [ ] **Largest Lyapunov exponent (Rosenstein/Wolf)** — recommended. Sequences are short (T=65); declare it as a limitation. Distribution of $\lambda_1$ across real vs generated.
- [ ] Takens embedding + correlation dimension (skip unless time permits).
- [ ] Bifurcation-style sweep along a high-variance latent direction (the controlled-generation interpretability story; partial work in `explore_oscillation_extrapolation.py`).

End-of-phase: at least one statistical comparison real vs generated on a standard NLD metric; one new figure or panel; PROJECT.md update.

---

## Phase 4 — First full draft

Goal: `paper.tex` compiles with all main figures inserted, ~6000–8000 words.

- [ ] **Methods** (rewrite, do not reuse stale `METHODOLOGY_DOCUMENT.md` as-is): data pipeline, scale-conditioned architecture, training schedules, extinction post-processing.
- [ ] **Results**: read every number from `RESULTS.json`. Subsections: reconstruction, max-val prediction, latent structure (PCA, t-SNE, active dims), LV adherence, interpretability + parameter recoverability (the new headline), baseline comparison, chaos diagnostics.
- [ ] **Discussion**: what worked, what didn't, the phenotypes-vs-parameters framing, limitations (short sequences, ≈5% near-extinction failures, slight train-distribution bias, no explicit physics constraint), future work.
- [ ] **Introduction** (write last): why generative models for ecological dynamics matter, why VAEs are appealing here, why scale is the hard problem, our contribution, why CSF.
- [ ] **Abstract** (write last). 150–250 words.

---

## Phase 5 — Supplement + reviewer pre-mortem

- [ ] Supplement: full latent-collapse analysis, 50D-vs-30D ablation table, threshold sweep, extra t-SNE/UMAP, per-curve recon panels, failure-mode panel.
- [ ] Write a 1-page "anticipated objections" doc. For each: do we have an answer? If not, add a paragraph.
- [ ] Cover letter draft.
- [ ] Have one friendly reader (PI / labmate) read cold and report what's confusing.

---

## Phase 6 — Co-author / advisor review

- [ ] Send full draft + supplement to advisor and co-authors. Give them a full week minimum.
- [ ] During wait: polish figures (consistent fonts, A/B/C labels, colorbar units, journal style), write data/code availability statement, double-check citations.
- [ ] **No new experiments this phase.**

---

## Phase 7 — Polish + submit

- [ ] Address co-author feedback.
- [ ] Final figure rev: vector PDFs, embedded fonts, 300 dpi rasters where vector isn't feasible.
- [ ] Format to Elsevier `elsarticle` template.
- [ ] Compile BibTeX with journal style.
- [ ] Cover letter, highlights bullets, graphical abstract, conflict-of-interest, data availability, suggested reviewers.
- [ ] Submit.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Baseline VAE matches conditioned on max-val R² | Architecturally implausible (max-val info is removed by normalization). If it happens, reframe around interpretability + controlled generation. |
| ~~GLV parameter recoverability comes back near 0~~ → addressed: it came back at 0.24 / 0.20 / 0.03 — *partial recovery*, the dominant-species story. |
| Chaos diagnostics show real and generated are distinguishable | Honest reporting; "model captures macroscopic dynamics but not full attractor structure." Still publishable. |
| Novelty/coverage shows memorization | Add rejection-sampling post-processing or KDE-based importance sampling on the prior. Don't retrain. |
| Compute fails / GPU unavailable | Most experiments are <1 GPU-hour. Baseline VAE is 3 hr; schedule early in Phase 2. |
| Co-author returns feedback late | Soft target ≈ mid-July leaves ~2-week buffer to hard target end-of-July. |

---

## Anti-goals (do NOT do these)

- Train a new model architecture (transformer, KAN-VAE, etc.).
- Add a new dataset (larger species count, longer sequences).
- Implement physics-informed loss terms.
- Refactor the codebase beyond what Phase 1 already did.
- Add a new evaluation metric every week. Use what's in `RESULTS.json`.

---

## Future-work bucket (deferred — for v2 or a follow-up paper)

- Physics-informed losses (GLV residual penalty).
- KAN-VAE, transformer, latent-ODE, neural-ODE baselines.
- Long-horizon extrapolation (T=65 → T=200).
- Active learning for under-represented regimes.
- β-VAE / DCI / MIG disentanglement metrics.
- Hierarchical CVAE with explicit interaction-matrix prior.
- Conditioning on initial conditions instead of (or in addition to) max values.
