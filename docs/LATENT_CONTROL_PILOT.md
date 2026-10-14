# Latent-control CPU pilot: promising, not physical validation

## Question

Can the existing model generate trajectories with requested overshoot and late-time
level without a visibly sorted latent space? This returns to the original
controllable-generation objective; noise/dynamics diagnostics are supporting
validity checks, not a replacement research objective.

## Scope and definitions

The pilot uses `model_final_30_conditioned.pth`, the documented original
**peak-sorted** 30D LSTM-VAE. It does not use the later no-sort checkpoints with
the scale-head mismatch, and no model was retrained. Sorted channel ranks are not
fixed biological species identities.

- Decode `z` on CPU, multiply normalized trajectories by predicted relative
  species factors, and take the community mean across species.
- **Tail level:** mean of the final 10 community observations in relative-family
  units. It is not a verified equilibrium or an absolute population abundance.
- **Overshoot:** post-initial peak excess relative to the tail level, counted only
  if the community starts below that level. Initial-condition decay is not called
  overshoot; a separate peak-excess diagnostic is retained.
- Require finite/nonnegative outputs, non-negligible peak and tail, and tail
  CV/drift no greater than 5%. This is an aggregate stationarity screen, not proof
  of equilibrium; opposing species trends can cancel in a community average.
- Predicted species factors weight the aggregate phenotype. This is community
  control, not independent control of each species' shape.

There is no external decoder condition: the original scale-conditioning pathway
enters the encoder. Both decoded shapes and relative factors are functions of z.

## Fixed pilot protocol

1. Draw 512 latent codes from the standard normal restricted to a fixed latent ball.
2. Split them: 256 provide the search bank/incumbents; the other 256 define target
   quantiles, IQR scaling, and a fixed-alpha ridge property probe.
3. Evaluate the property probe on 256 fresh latent codes. Select any displayed
   single axis using calibration only, not the probe set.
4. Request the Cartesian product of the 20th/50th/80th control percentiles. Keep
   every distinct joint target, including sparse or potentially incompatible corners.
5. Compare a zero-online-cost bank lookup with two arms sharing the same incumbent:
   64 fresh random proposals, or two rounds of 32 local proposals around eight
   nearest codes in standardized control space. Both arms obey the same latent-ball
   constraint via rejection, and invalid/nonstationary decodes consume the budget.
6. Repeat each target with three independent proposal RNG streams. These are search
   repetitions, **not** three independently trained models or independent target banks.

Success requires coordinatewise errors <= 0.15 calibration IQR, using
`log1p(overshoot)` and linear relative tail level. Thresholds/search settings were
fixed before inspecting the actual pilot results. A collapsed control IQR produces
an explicit negative result rather than artificial control on a rank scale.

The noisy real reference is range context only; it sets no targets, fit, thresholds,
or search settings. Its observation noise is not matched to decoder mean outputs,
so overlapping points are not a distributional-fidelity result.

## Observed result

Local run: `results/runs/latent-control-cpu-20261014T225431510533Z/`.
The directory contains `report.json`, `samples.npz`, `control_feasibility.png`, and
`targeted_trajectories.png`. It remains ignored local exploratory output. The report
records checkpoint/reference hashes, source hashes, parameters, sample indices,
candidate budgets, per-target errors, and novelty/diversity diagnostics.

| Measure | Result |
|---|---:|
| Valid calibration decodes | 512 / 512 |
| Calibration decodes passing the aggregate stationarity screen | 507 / 512 |
| Fresh-code ridge R²: log1p overshoot | 0.559 |
| Fresh-code ridge R²: relative tail level | 0.934 |
| Bank-only distinct targets attained | 8 / 9 |
| Random + bank target trials attained | 24 / 27 |
| Local search + bank target trials attained | 27 / 27 |
| Median normalized error, random + bank | 0.0333 |
| Median normalized error, local + bank | 0.0136 |
| Paired trials: local wins / ties / losses | 21 / 4 / 2 |
| Local selections that improve on the stored bank incumbent | 23 / 27 |

The requested overshoot levels were approximately 0.005, 0.059, and 0.235; relative
tail levels were approximately 0.302, 0.372, and 0.448. These are interior requests,
not extreme extrapolation. The sparse upper-right combination had no bank sample
within tolerance; local search attained it in all three repetitions.

The median-error reduction is approximately 59%, but bank lookup already solves
most requests. The incremental finding is improved precision and one sparse joint
corner, not that random sampling contains no useful controls. Decoder repeatability
was exact for the eight checked codes.

The run decoded 4,240 trajectories: 512 bank/definition samples, 256 probe samples,
16 repeatability checks, and 1,728 online samples per arm. Measured decoder time
was approximately 4.37 seconds on one CPU thread. That is a local measurement,
not a speed comparison with ODE integration or a training-cost claim.

A supplementary check of the saved bank found a median decoded family peak of
0.962, with 97.5% within 10% of 1. The decoder's coordinate system is therefore
close to, but not exactly, actual-peak normalization. Explicitly normalizing each
generated family changes tail level but leaves overshoot invariant. Include that
normalization sensitivity before interpreting the tail-level control physically;
it was not used to tune or rerun the reported search.

## What this does and does not establish

**Supported within this pilot:** there is usable latent/property structure, and
target-guided sampling can be useful without neat visual or coordinate ordering.
The original sorted model is worth investigating before committing to many reruns.

**Not established:** controlled true GLV equilibria, validity of individual species
trajectories, preservation of interaction matrices, diversity beyond small local
variations, novelty relative to training data, generalization across checkpoints,
or advantage over direct simulator rejection/parameter sampling. A low linear
probe R² would also not disprove nonlinear controllability.

Next, inspect individual species and use an independently justified GLV validity
test; compare target-conditioned diversity and simulator-based baselines. Test
broader independent target sets rather than optimizing the current nine cases.
Do not expand the architecture count merely because the latent axes look mixed.

## Reproduction

```bash
TimeSeries/bin/python analysis/latent_control_feasibility.py
```

The script hides CUDA before numerical imports, limits numerical threads to one,
loads the checkpoint on CPU, never trains the VAE, and creates a new output
directory rather than overwriting any existing result or figure. Reference pickle
loading assumes trusted local repository data. Use `scripts/run_cpu_checks.py`
for the small dependency-optional fixture tests without loading the checkpoint.
