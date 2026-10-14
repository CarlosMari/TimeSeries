# Corrected scale contract: CPU implementation, not retrained evidence

The scale-correctness implementation is available for LSTM-VAE, Transformer-VAE,
and the LSTM learned-variance variant. **No corrected model has been trained or
evaluated on the research datasets.** Existing results remain historical.

## Explicit compatibility boundary

- `legacy_sorted_v1`: the old C-1-output head, fixed species-0 factor, and linear
  scale MSE over species 1 onward. Constructors default to this for compatibility
  with old scripts/checkpoints. This name describes the intended sorted-data
  contract; historical no-sort checkpoints remain scientifically confounded.
- `species_v2`: C outputs with no fixed dominant species, all-species scoring,
  and centered log-ratio loss. The three selected model families support it.
  Stochastic-hidden-state, latent-ODE, and KAN variants are not upgraded study arms.

`train_pivot.py` now **requires** `--scale-contract`. Historical queue commands
without it stop at argument parsing; they are not automatically upgraded or run.
Legacy training on unsorted data is rejected unless the explicitly historical
`--allow-legacy-on-unsorted` override is supplied. Existing checkpoint paths are
not overwritten by the pivot CLI. `--help` never starts training.

## Geometry, loss, and limits

For predicted logits `l`, the corrected relative factors are
`exp(l - max(l))`. They are positive, at most 1, and permit any species to be
dominant. A machine-range floor avoids exponentiation underflow.

Training uses `c = l - mean(l)` against
`log(max(target, scale_log_floor)) - mean(log(max(target, scale_log_floor)))`.
This is a smooth centered log-ratio MSE computed in fp32; it avoids training on a
max-normalized coordinate whose currently dominant component is fixed at zero.
The transformed output in the forward tuple therefore means centered logits for
v2, not the legacy raw transformed partial factors.

The default target log floor is `1e-8`, recorded in the configuration. It is
numerical stabilization, **not** an estimated ecological detection threshold.
Targets must be finite, strictly positive, no greater than 1 within tolerance,
and have a maximum of 1. All-zero families and exactly absent species are rejected
rather than assigned an invented dominant species. Below-floor targets are counted
in corrected evaluation metrics. The controlled generator's clean peak/abundance
selection also excludes vanishing species peaks under its default settings.

The training dataset guard checks the normalized curves too, in bounded chunks:
the old preprocessor's `1e-8` replacement for a zero peak is not treated as evidence
that the species is present. Metrics count targets **at or below** the numerical
floor with dtype tolerance. Per-species R² is reported as `null` when the target
is constant, rather than a huge meaningless negative value. Pooled R² can reflect
between-species differences, so it is not the primary scale-fidelity claim.

The conditioning encoder still receives **linear** relative factors. The absolute
family maximum is a separate, unresolved generation component. Since a decoded
mean curve need not attain unit peak, the head output is a relative amplitude
factor; it is not a guarantee that a sampled observation has that exact peak.

**The scale-loss units changed.** Do not reuse the old weight as a tuned value,
attribute all changes to the seventh output alone, or compare old and corrected
training losses as the same objective. Validate effective scale/KL weights before
the production experiment.

## Checkpoint and evaluation safety

Corrected saves use checkpoint format version 2, architecture identity, a resolved
model configuration, a state dictionary, and supplied training metadata. Only
plain types and tensors are stored, supporting `weights_only=True` loading.
Plain state dictionaries require an explicit legacy loader configuration and are
never interpreted as corrected models. Six- and seven-output heads do not load
into each other, even with `strict=False`.

Corrected metadata is validated before logging, device transfer, or fitting.
Corrected saves are atomic and refuse existing final files. An initial and
periodic `.progress.pth` snapshot protects completed weights if evaluation or
serialization fails later. These are weight-recovery snapshots, not optimizer/RNG
state for exact training resumption. No such production snapshots have been made
in this CPU-only implementation phase. Legacy evaluation retains its old result
schema and scale-score inclusion rule.

The unified evaluator's selected-model adapters accept both formats. Corrected
scale metrics include all species, per-species and pooled R², linear MSE, centered
log-ratio error, dominant-species accuracy (allowing tied true maxima), and
below-floor counts. Corrected checkpoints cannot write into `results/legacy/`.

**The remainder of the old evaluator is still historical:** it borrows test
family scales, applies extinction post-processing, and uses the unvalidated
short-window divergence protocol. Supporting v2 checkpoint loading does not make
that evaluator the approved final controlled experiment.

## Safe checks while the GPU is reserved

```bash
TimeSeries/bin/python scripts/run_cpu_checks.py
```

The runner hides CUDA before scientific imports, caps numerical thread counts at
1, disables W&B logging, and runs tiny temporary-fixture tests only. Model tests
also guard against CUDA initialization. With a stdlib-only interpreter, numerical
tests are skipped explicitly rather than importing missing dependencies.

Tests cover every dominant-species index, normalization round trips, gauge and
permutation consistency of the scale helper (not equivariance of the whole model),
wrong-dominant gradients, legacy formulas, dataset guards, CPU forward/backward and
generation, all-species scoring, checkpoint round trips, and adapter loading.
The initial scale-contract block passed 79 checks. The CPU runner also includes
subsequently added control-pilot fixtures; a stdlib-only run explicitly skips
optional numerical tests rather than treating missing packages as success.

A separate tiny CPU comparison against pre-change commit `a5cce18` verified exact
legacy state keys/shapes, forward/generation outputs for all three selected
families, and the legacy VAE loss. This is a compatibility check, not a trained
performance experiment; CUDA/AMP has not been tested.

Local trusted-pickle compatibility checks also found:

| Dataset | N | Species 0 is dominant | Smallest relative peak | Exactly absent species |
|---|---:|---:|---:|---:|
| TRAIN_FINAL_PROCESSED | 117472 | 100% | 0.03351960 | 0 |
| TEST_FINAL_PROCESSED | 39189 | 100% | 0.03352349 | 0 |
| TRAIN_FINAL_NOSORT | 117472 | 14.2238% | 0.03351960 | 0 |
| TEST_FINAL_NOSORT | 39189 | 14.5398% | 0.03352349 | 0 |

Thus the old sorted files satisfy the legacy guard even without sort metadata;
the no-sort files do not. These session-local checks establish input compatibility,
not training provenance or new paper metrics. No dataset was rewritten.

## Matched-simulator foundation

`data_generation/controlled_glv.py` adds an independently tested single-system
generator with fixed physical time, recorded solver/parameter settings, separate
parameter and observation-noise seeds, paired clean/noisy trajectories, clean-only
selection, explicit failures, and grouped train/validation/test assignment.

It uses domain-separated local PCG64/SeedSequence generators (including when the
numeric parameter and noise seeds coincide) and explicit tolerances, so it is a **new data
version**, not a bitwise regeneration of the old RandomState-based datasets.
Strict feasible-equilibrium stability is a sampling rule, not proof that a whole
trajectory is nonchaotic. Solver-work limits are deterministic evaluation caps,
not wall-clock guarantees. No production dataset builder or data files have been
created yet. Hash-based split fractions are probabilities, not exact sample quotas.
Even an isolated growth-scale change can alter clean-selection acceptance. Any
future OOD report must retain rejection/attrition rates rather than presenting the
accepted sets as an unfiltered parameter-only intervention.

`configs/controlled_study_v2.proposed.json` is a reviewable proposed contract, not
an executable training configuration. Dataset size, loss weights, sampling policy,
and final diagnostic settings still need decisions. No training or production
generation is scheduled, and GPU use remains disabled for this work phase.
