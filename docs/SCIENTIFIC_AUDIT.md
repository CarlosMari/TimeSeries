# Scientific audit: boundaries on current claims

October 2026 review. This records issues found in existing code and stored
artifacts; **the cleanup does not repair them or recompute results**. Read this
before using the historical project narratives in a manuscript.

**Follow-up status:** the all-seven-species scale contract, associated guards, and
single-system matched-simulator foundation are now implemented and CPU-tested.
No corrected model has been fitted and no historical result was recomputed.
[CORRECTED_SCALE_CONTRACT.md](CORRECTED_SCALE_CONTRACT.md) distinguishes this code
progress from the unresolved controlled study and the issues below.

## 1. No-sort data versus a fixed species-0 scale head

`data_generation/preprocessor.py` preserves original species order when
`sort_curves=False`. The VAE-family scale heads still predict only six ratios and
prepend 1 for species 0 (`src/models/cvae.py` and the related model classes).
`train_cvae.vae_loss` and comparative scale scoring omit species 0.

In the locally inspected `TEST_FINAL_NOSORT.pkl`, species 0 is the largest in
approximately 14.5% of trajectories; its mean relative peak is approximately
0.526. This contradicts the fixed-1 assumption. Resolve the representation for
all seven species, test it, then retrain affected models. A new evaluation alone
cannot retroactively correct these checkpoints.

These are session-local exploratory calculations, not newly certified paper
metrics: the first quantity is the mean of
`isclose(reconstruction_max_values[:, 0], 1)` and the second is the mean of
`reconstruction_max_values[:, 0]` in the trusted local pickle. The inspected file's
SHA256 was `161da0564ad4b0d57e9d7b3ae5b1e7d1cc3b1d67c76b72624a5fd6c4a1c6d3cd`.
This identifies the inspected bytes, not their training provenance.

## 2. The current OOD shift is compound

The ID generator uses fixed `tmax=20`, multiplicative lognormal observation noise,
and abundance/overshoot rejection filters. `data_generation/generate_ood.py`
instead uses an eigenvalue-dependent horizon, omits that observation noise, and
omits those trajectory-selection filters. Therefore existing OOD differences
cannot be attributed to growth-rate distribution alone.

The scripts use NumPy's exponential **scale**, not a rate parameter. Document the
parameterization explicitly; some historical faster/slower labels are reversed.
Generation is unconditional, so comparing one fixed generator to OOD observations
does not demonstrate adaptation to an OOD regime. The evaluator also samples the
global scale from the test set: disclose this empirical component of generation.

## 3. Short-window divergence is not validated chaos recovery

`analysis/chaos_diagnostics.py` embeds a species-averaged 65-point signal in three
dimensions with delay 2, then fits early log-distance growth per observation step.
Noise, transients, aggregation, embedding, and sampling interval can affect this
statistic. Positive values alone are not evidence of a positive physical
Lyapunov exponent.

Exploratory checks using this exact estimator found positive values for IID noise,
periodic signals, and noisy stable relaxation. These are a warning about
interpretation, not a calibrated validation study. Either validate against known
dynamics and simulator-based exponents (including physical time and appropriate
horizons), or report a short-window divergence statistic without chaos claims.
High RQA determinism likewise does not establish a unique attractor or mode collapse.

The checks were session-local and have no persisted metric report; do not copy
their exploratory rates into a manuscript as validated results. The inspected
`analysis/chaos_diagnostics.py` SHA256 was
`49e6c981c6f47aa32354609cffa30d1bf7e5de598e383f0dee0b009668595d18`.

## 4. Learned variance versus observation sampling

Existing baseline `.generate()` paths return decoder means; M8 samples from its
learned variance head. A matched fixed-variance sampled baseline is required to
isolate the value of heteroscedasticity. M8 also changes the reconstruction
objective and feeds means during training but samples during generation.

`results/legacy/RESULTS_M8_FINAL_NOSORT.json` shows generated mean DET of approximately 0.572,
0.416, and 0.454 across its three seeds, rather than a uniform match to the real
mean of approximately 0.601. It also has lower reconstruction R² and higher
feature MMD than the saved Transformer results. Average agreement in one statistic
is not an architectural cure or recovery of physical dynamics.

## 5. Post-processing and statistics

- The evaluator applies irreversible extinction thresholding to generated samples
  but not the real observations. Report raw generation and an explicit ablation;
  low positive GLV abundance is not automatically irreversible extinction.
- A nonsignificant KS test is not an equivalence test. Avoid calling it proof of
  statistical indistinguishability.
- Extremely small p-values need effect sizes and real-versus-real reference
  variation, not a universal "more significant is better" ranking.
- Distinguish variability across generated trajectories from uncertainty across
  training seeds; many generated samples do not substitute for independent runs.
- Match standardization/kernel settings for comparable feature distances. The
  main evaluator and noise sweep currently fit feature scaling differently.
- Preserve selection and failure rates. Silent imputation of non-finite features
  or exclusion of non-finite diagnostic values must not hide failed samples.

## Status of the literature and release

Combining existing MMD, recurrence, and divergence diagnostics is not automatically
a novel method. Check the closest scientific time-series generation/evaluation
work before fixing the contribution. `REFERENCES.md` includes incomplete entries
and conceptual interpretations that still require verification. Historical venue
confidence statements are not evidence of publishability.
