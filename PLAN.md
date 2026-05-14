# PLAN.md — Path to *Chaos, Solitons & Fractals* in ≤ 8 weeks

**Today:** 2026-05-14  **Hard target submission:** 2026-07-09 (8 weeks)
**Soft target submission:** 2026-06-25 (6 weeks; leaves a 2-week buffer for revisions, co-author reads, formatting).

This plan is built around PROJECT.md. Read that first — it has the verified metrics and a frank list of what's solid vs soft. This document tells us *what to do next, in what order, by when*.

---

## Guiding principles for the next 8 weeks

1. **No new architecture.** The 30D scale-conditioned CVAE is the model. Resist the urge to retrain. Every "tiny model tweak" eats a week.
2. **Fix things that are wrong before adding things that are missing.** The LV-validation bug, the stale methods text, and the figure inventory are the highest-leverage uses of time because they affect *everything else*.
3. **Reviewer-driven scope.** Pretend a *Chaos, Solitons & Fractals* reviewer is reading. Their three predictable objections:
   - "Compare to a baseline."
   - "What about chaos-specific diagnostics (Lyapunov, fractal dim, recurrence quantification)?"
   - "Is the model just memorizing the training distribution?"
   Each one needs a panel or paragraph. Plan around them.
4. **Time-box.** If something hasn't yielded a usable result in 2 working days, cut it and document why.

---

## Critical-path summary

```
W1  Fix-LV-bug + corrected numbers + figure audit            ──┐
W2  GLV-parameter recoverability + baseline VAE training       ├─ Results locked
W3  Chaos diagnostics + novelty/coverage                     ──┘
W4  First full draft (methods + results)                     ──┐
W5  Discussion + intro + ablations + supplement                ├─ Manuscript locked
W6  Co-author / advisor review iterations                    ──┘
W7  Final figures pass + formatting + cover letter           ──┐
W8  Submission, buffer for last-minute findings              ──┘
```

Anything not listed below is out of scope for v1. Capture it in a `FUTURE_WORK.md` (or just a markdown checklist at the end of this file) and move on.

---

## Week 1 — Lock down the numbers (May 14 → May 21)

Two big items: fix the embarrassing LV bug and produce a clean, single source of truth for every paper number.

### 1.1 LV-adherence pipeline — rewrite from scratch ⚠️
- [ ] Write one canonical function `lv_adherence_r2(trajectory: np.ndarray, dt: float = 1.0) -> dict` that returns mean / median / per-species R² and number of near-zero samples.
- [ ] Use `np.gradient` *or* finite differences consistently; document which.
- [ ] Apply extinction threshold θ = 0.005 (winner from the sweep) as the default.
- [ ] Run on (a) 5000 real test samples, (b) 5000 generated raw, (c) 5000 generated + extinction fix.
- [ ] Bootstrap 1000× to get 95% CIs on the mean and the % > 0.9.
- [x] Re-generate `fig_lotka_volterra_validation_with_fix.pdf` with corrected numbers (done 2026-05-14: real = 0.9875 mean LV-R², gen with θ=0.005 = 0.9680).
- [x] Retire `LV_VALIDATION_SUMMARY.md` / `RESULTS_TEXT_LV_VALIDATION.txt` (deleted 2026-05-14).

### 1.2 Single results-table source
- [x] One script `analysis/produce_paper_metrics.py` that loads the 30D checkpoint, runs every metric (recon, max-val per-curve, latent variance, PCA, LV-R², novelty), and writes `RESULTS.json` + `RESULTS.md`. **This is the only place numbers live.** Every figure script reads from this. (Done 2026-05-14.)
- [x] Bootstrap 95% CIs on recon R², max-val R², LV R². (Done — see RESULTS.md.)

### 1.3 Figure audit + cleanup
- [ ] List exactly which 8 main figures + 6 supplement figures will appear in the paper (PROJECT.md §5 is the starting point).
- [ ] Regenerate any 50D-era or non-conditioned figures with the 30D conditioned model.
- [ ] **`fig_max_value_prediction.pdf` must be regenerated** — current version reports the pre-conditioning, negative-R² result.
- [ ] Move stale figures to `experiments/archived/figures_pre_conditioned/`.

### 1.4 Repo hygiene (boring but necessary)
- [x] Picked `train_cvae.py` at the root; deleted `src/training/train_cvae.py`.
- [x] Fixed β-schedule warmup line in `train_cvae.py`.
- [x] Tightened `.gitignore` (LaTeX intermediates, root-level ad-hoc PNG/PKL, removed bad `utils/` exclusion).
- [ ] Move all top-level `generate_*.py` into `analysis/`. **Deferred** — risk of breaking working figure pipelines before submission. Revisit after first draft compiles.

**Week 1 success criterion:** running `python analysis/produce_paper_metrics.py` reproduces every headline number with CIs, and `final figures/` contains only paper-bound, current artifacts.

---

## Week 2 — Strengthen the science (May 21 → May 28)

Two new experiments. Each is ≤2 days. Either can be cut if it doesn't yield.

### 2.1 GLV parameter recoverability (1.5 days)
The most natural reviewer question: "Does the latent space know anything about $(r, A)$?"

- [ ] For 5000 train + 5000 test trajectories whose $(r, A)$ we generated, encode to get $\mu(z) \in \mathbb{R}^{30}$.
- [ ] Fit Ridge regression $\mu \to r$ (7 outputs) and $\mu \to \mathrm{vec}(A)$ (49 outputs) on train, evaluate on test.
- [ ] Report per-target R². Either outcome is publishable:
  - Strong: "latent space recovers GLV parameters with $R^2 = X$ — implicit system identification."
  - Weak: "latent space encodes phenotypes, not parameters (see §interpretability)."
- [ ] One figure: scatter `true vs predicted` for $r$ and the strongest column of $A$.

> Need `(r, A)` for every saved sample. Check whether they were stored at generation time; if not, this is more work and might slip to week 3.

### 2.2 Baseline comparison: vanilla VAE without scale conditioning (2 days)
- [ ] Train one 30D non-conditioned LSTM-VAE with otherwise identical config for 2000 epochs (≈3 hr).
- [ ] Compute the same battery of metrics with the *same* `produce_paper_metrics.py` script.
- [ ] One table panel: conditioned vs non-conditioned on recon R², max-val R², LV R², latent active dims.
- [ ] Bonus baseline if time: identical-config model trained on un-sorted curves (to motivate the curve-sorting step).

### 2.3 Novelty / coverage — proper evaluation (0.5 day)
The 0.94 memorization ratio in PROJECT.md §3.5 needs a real treatment, not a footnote.

- [ ] Compute density-coverage scores (Naeem et al. 2020) on a feature-vector representation of trajectories.
- [ ] Or simpler: 2-sample MMD test between generated and held-out test in feature space.
- [ ] Goal: either show "generated and held-out are statistically indistinguishable in feature space" or quantify the gap honestly.

**Week 2 success criterion:** a Table 1 draft with conditioned vs baseline numbers, and a clear yes/no on parameter recoverability.

---

## Week 3 — Chaos-journal-specific analyses (May 28 → June 4)

The journal is *Chaos, Solitons & Fractals*. We need at least two nonlinear-dynamics analyses beyond the recurrence plots we already have. Pick **two** of these — don't try all four.

### 3.1 Recurrence Quantification Analysis (RQA) — recommended ⭐
- [ ] For matched real and generated samples (n=200 each), compute recurrence rate, determinism, average diagonal line length, laminarity, trapping time.
- [ ] Use `pyts.image.RecurrencePlot` + manual quantification, or `pyRQA`.
- [ ] Table: real vs generated, with a statistical test (KS or MMD per metric).
- [ ] Extend the existing `fig_recurrence_dynamics.pdf` to include the quantification panel.

### 3.2 Largest Lyapunov exponent estimation — recommended ⭐
- [ ] Rosenstein or Wolf algorithm on the 65-step trajectories. Yes, sequences are short; that's a *known limitation* we declare.
- [ ] Distribution of $\lambda_1$ across real vs generated. Does the model preserve chaotic vs regular regimes?

### 3.3 Phase-space topology
- [ ] Takens-embedding of single-species trajectories, compute correlation dimension and entropy.
- [ ] Compare distributions real vs generated.

### 3.4 Bifurcation / attractor structure
- [ ] Sweep a single latent dimension (one of the high-variance ones) and observe phase-portrait evolution.
- [ ] Closes the "controlled generation" story with an honest dynamical-systems framing.

**Week 3 success criterion:** two new chaos-specific results integrated into figures, with at least one statistical comparison real vs generated.

---

## Week 4 — First full draft (June 4 → June 11)

Goal: end of week, a complete `paper.tex` that compiles, ~6,000–8,000 words, with all paper-bound figures inserted.

### Sections, in writing order
1. **Methods** (write first — it's the most mechanical). Source: `METHODOLOGY_DOCUMENT.md` + `CODEBASE_REFERENCE.md`. **Rewrite to match the 30D conditioned architecture.** Subsections:
   - Data: GLV generation, the 3-stage normalization, train/test split.
   - Model: shape encoder, scale encoder, bottleneck, max-value head, decoder.
   - Loss and training schedules.
   - Extinction post-processing.
2. **Results** (use `produce_paper_metrics.py` outputs verbatim):
   - Reconstruction quality.
   - Max-value prediction (the $R^2 = 0.97$ headline).
   - Latent-space structure (PCA, t-SNE, active dims).
   - LV adherence (corrected numbers).
   - Interpretability (the dim ↔ feature breakdown).
   - Baseline comparison.
   - Chaos diagnostics from week 3.
3. **Discussion**:
   - What worked, what didn't.
   - The "phenotypes vs parameters" story for the latent space.
   - Limitations: short sequences, 5% near-extinction failures, slight train-distribution bias of generated samples, no explicit physics constraint.
   - Future work.
4. **Introduction** (write last — easier when everything else is in place). Story arc:
   - Why generative models for ecological dynamics matter.
   - Why VAEs are appealing here (latent space → interpretability + control).
   - Why the scale problem makes it hard.
   - Our contribution: dual-encoder scale conditioning.
   - Why *Chaos, Solitons & Fractals*: phase-space and recurrence framing, GLV is a canonical nonlinear system.
5. **Abstract** (last). 150–250 words. Lead with the architectural innovation, headline number, the LV-adherence result, and the interpretability story.

### Deliverables end of week 4
- [ ] `paper.tex` compiles to a PDF.
- [ ] All 8 main figures inserted with captions.
- [ ] Reference list with at least 30 citations (LSMGD, β-VAE, conditional VAEs, GLV theory, recurrence quantification, etc.).

---

## Week 5 — Refine, add supplement (June 11 → June 18)

### 5.1 Supplementary material
- [ ] Full latent-collapse analysis (variance per dim, all 30 dims).
- [ ] 50D-vs-30D ablation (the table from `MODEL_COMPARISON_30D_vs_50D.md`).
- [ ] Extinction-threshold sweep figure.
- [ ] Extra t-SNE/UMAP at varying perplexity.
- [ ] Reconstruction-quality scatter plots, per-curve panels.
- [ ] Failure-mode panel (5% R² < 0.7 cases).

### 5.2 Reviewer pre-mortem
- [ ] Write a 1-page "anticipated objections" document. For each: do we have an answer in the paper? If not, write a paragraph that pre-empts it.
- [ ] Have one friendly internal reader (PI or labmate) read the draft cold and report what's confusing.

### 5.3 Cover letter draft

---

## Week 6 — Co-author / advisor iteration (June 18 → June 25)

- [ ] Send full draft + supplement to advisor and any co-authors **by start of week 6**. They get a full week.
- [ ] Use the wait time to: polish figures (consistent fonts, panel labels A/B/C, colorbar units, journal style), write the data/code availability statement, double-check citations.
- [ ] Don't start new experiments this week. Things added in week 6 don't get reviewed.

**Soft submission target:** end of week 6 (June 25) if everyone says yes.

---

## Weeks 7–8 — Polish + submit (June 25 → July 9)

- [ ] Address co-author feedback.
- [ ] Final figure rev: vector PDFs, embedded fonts, consistent 300 dpi rasters where vector isn't feasible.
- [ ] Format to Elsevier's `elsarticle` template.
- [ ] Compile reference list with the journal's BibTeX style.
- [ ] Prepare the submission checklist (cover letter, highlights bullets, graphical abstract, conflict-of-interest, data availability, suggested reviewers).
- [ ] Submit.

**Hard submission:** July 9. If we miss this, the next acceptable date is the first week of September (after the typical summer slowdown), which is outside the stated 2-month window. Don't miss.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Re-running LV validation reveals worse numbers than expected | Already done in PROJECT.md — numbers are 0.95 vs 0.97, still strong. No surprise risk. |
| Baseline VAE matches conditioned on max-val R² | Architecturally implausible (max-val info is removed by normalization), but if it happens, the paper reframes around interpretability + controlled generation rather than scale recovery. |
| GLV parameter recoverability comes back near 0 | Becomes a *result*: the latent space encodes phenotypes, not parameters. Already the interpretability story in PROJECT.md §4. |
| Chaos diagnostics show real and generated are statistically distinguishable | Honest reporting; framed as "model captures macroscopic dynamics but not full attractor structure." Still publishable. |
| Novelty / coverage analysis shows real memorization | Add rejection-sampling post-processing (easy) or KDE-based importance sampling on the prior. Don't retrain. |
| Compute fails / GPU unavailable | Most experiments are <1 GPU-hour. Train one baseline is 3 hr. Schedule it for week 2 day 1 to leave slack. |
| Co-author returns feedback late | Soft target = end of week 6 leaves 2 full weeks of buffer. |

---

## Anti-goals (do NOT do these)

- Train a new model architecture (transformer, KAN-VAE, etc.). Not in scope.
- Add a new dataset (larger species count, longer sequences). Out of scope.
- Implement physics-informed loss terms. Future work, not v1.
- Refactor the codebase beyond §1.4 of week 1. Cleanup is good; refactoring eats months.
- Add a new evaluation metric every week. Pick the metrics in week 1 and use them everywhere.

---

## What "done" looks like, week by week

| End of week | Deliverable |
|---|---|
| 1 | `RESULTS.json` + `produce_paper_metrics.py` + clean figure directory |
| 2 | Baseline-vs-conditioned table; GLV recoverability result |
| 3 | Two chaos-specific analyses with figures |
| 4 | Full draft compiles, all main figures inserted |
| 5 | Supplement done; reviewer pre-mortem written |
| 6 | Draft out to co-authors |
| 7 | Final revisions complete, formatted to Elsevier style |
| 8 | **Submitted.** |

---

## Daily ritual

Each work session, append three lines to a `LOG.md`:

```
2026-MM-DD
  did: <one sentence>
  blocked on: <one sentence or "nothing">
  next: <one sentence>
```

Cheap, honest, prevents 80% of the lost-context grief that's already shown up in the code/doc divergence.

---

## Future-work bucket (deferred — not for v1)

- Physics-informed losses (Lagrangian / GLV residual penalty).
- KAN-VAE, transformer, diffusion-based alternatives for comparison.
- Long-horizon extrapolation (training on T=65, decoding to T=200).
- Active learning for under-represented regimes (the ultra-oscillatory extrapolation work).
- $\beta$-VAE / disentanglement metrics (MIG, DCI).
- Hierarchical CVAE with explicit interaction-matrix prior.
- Conditioning on initial conditions instead of (or in addition to) max values.

Park these in a follow-up paper. v1 ships.
