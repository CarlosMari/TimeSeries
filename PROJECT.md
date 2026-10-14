# Project status

**Current review: October 2026.** This is the entry point for project status, not a
running transcript of every hypothesis. Historical May 2026 narratives and numbers
are preserved in [archive/docs/PROJECT_2026-05.md](archive/docs/PROJECT_2026-05.md).

## Scope

Study what reconstruction and sample-distribution diagnostics reveal about VAE
generation of Generalized Lotka–Volterra trajectories (7 species, 65 observations).
A candidate framing is separating observation-noise fidelity from underlying
trajectory fidelity. This is a research question, not a confirmed publication claim.

Implemented variants include scale-conditioned and unconditioned LSTM-VAEs,
hidden-state-noise variants, latent-ODE, Transformer-VAE, KAN-VAE, and a learned
variance-head LSTM-VAE (M8). The GLV-regression experiment has known integration
setup issues and is not a validated reference baseline.

## Evidence status

- Saved comparative metrics and checkpoints exist; see [results/README.md](results/README.md).
- Smooth generated samples and strong normalized reconstruction coexist in the
  saved results. Their physical interpretation is not settled.
- Removing species sorting did not remove the fixed-species-0 scale assumption.
- ID and OOD generators differ in noise, time horizon, and rejection filters.
- The short-window divergence estimate is noise-sensitive and is not established
  as a measure of chaos for these observations.
- M8 changes both the objective and observation sampling; it is not yet an
  isolated test of heteroscedasticity or a demonstrated dynamics "cure."
- The artifact manifest records what is present, not independently verified
  training provenance. There is no designated final paper result set.

The detailed issues and required controls are in
[docs/SCIENTIFIC_AUDIT.md](docs/SCIENTIFIC_AUDIT.md). Do not promote historical
"publication-ready," "perfect match," or "statistically indistinguishable" labels
to current conclusions without resolving those issues.

## Working agreements

1. Keep repository organization changes separate from scientific behavior changes.
2. Preserve old results, checkpoints, figures, and uncommitted work.
3. Use configuration and provenance, not a filename containing `FINAL`, to identify
   an experiment.
4. Add regression tests before modifying shared model or simulator assumptions.
5. Freeze architecture expansion until the matched-noise controls answer the
   central question.
6. Write alongside the controlled experiments; do not accumulate another unbounded
   figure or latent-space exploration phase.

## Navigation

- [PLAN.md](PLAN.md): remaining work and stopping criteria.
- [docs/REPRODUCING.md](docs/REPRODUCING.md): commands and environment constraints.
- [docs/CLEANUP.md](docs/CLEANUP.md): cleanup boundary and preserved compatibility.
- [REFERENCES.md](REFERENCES.md): citation working notes, including unverified placeholders.
- [Historical design](docs/superpowers/specs/2026-05-15-comparative-evaluation-design.md):
  motivation for the earlier comparison; not the current executable plan.
