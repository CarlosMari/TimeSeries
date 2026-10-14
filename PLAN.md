# Bounded plan to a publication decision

This replaces the historical rolling roadmap, preserved at
[archive/docs/PLAN_2026-05.md](archive/docs/PLAN_2026-05.md). No submission date or
venue acceptance is implied. Scientific corrections below are **not** implemented
by the repository cleanup.

## 1. Repository cleanup

- [x] Preserve historical scientific narratives instead of deleting them.
- [x] Separate exploratory root scripts from the supported entry points.
- [x] Document current evidence limitations and historical reproduction boundaries.
- [x] Declare the numerical pipeline dependencies, including `torchdiffeq`.
- [x] Add a standard-library artifact inventory and lightweight maintenance tests.
- [ ] Select final figure and result sets after the scientific decision gate.
- [ ] Publish or otherwise document retrieval of exact datasets/checkpoints.
- [ ] Resolve distribution license and vendored-code notices before release.

## 2. Scientific correctness gate

| Priority | Required change or experiment | Completion criterion |
|---|---|---|
| P0 | Correct scale representation for unsorted species | Automated test covers all seven species and no fixed dominant index |
| P0 | Use a shared simulator configuration for ID and OOD | Noise, time grid, selection filters, parameters, and seeds recorded explicitly |
| P0 | Isolate mean outputs from observation sampling | Fixed-noise, learned-noise, and mean-only outputs compared fairly |
| P0 | Audit raw versus extinction-thresholded generation | Primary conclusions do not silently depend on asymmetric post-processing |
| P1 | Validate diagnostic interpretation | Known nonchaotic controls and parameter sensitivity reported; chaos claims removed unless independently supported |
| P1 | Lock statistics before rerunning | Effect sizes, real-versus-real reference variation, and training-seed uncertainty reported |

Do not launch wholesale retraining until tests cover the corrected scale and data
contracts. Existing checkpoints remain historical evidence, not corrected models.

## 3. Small decisive study

Start with corrected LSTM-VAE, Transformer-VAE, and M8; keep other existing
variants as supporting material unless a specific control requires them. Retain
matched clean/noisy simulator realizations and distinguish observation fidelity
from underlying trajectory structure. Use a validation split for choosing noise
levels and settings, then a fresh held-out evaluation for the final comparison.

**Decision question:** after matching the observation model, do meaningful
trajectory-structure discrepancies remain?

- If yes: write the focused dynamics-fidelity case study.
- If mostly no: write the narrower evaluation/observation-model caution, if the
  literature check establishes a useful contribution.
- If fixed-noise sampling matches M8: report that result; do not force an
  architectural-novelty narrative.

Time-box this decision phase to approximately two working weeks after the three
authors agree on the experiment contract. Do not expand model count by default.

## 4. Parallel author responsibilities

- Implementation: correctness tests, shared simulator, matched comparisons, provenance.
- Dynamics: diagnostic validation, physical interpretation, claim boundaries.
- Writing/literature: closest prior work and a narrow draft while controls run.

Stop adding exploratory figures until the study has a fixed claim, a complete
draft, and a short list of evidence each main figure must provide.
