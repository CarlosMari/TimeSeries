# Next-work plan: one researcher, two advisors

**Owner:** Carlos implements the experiments, analyzes results, and writes the
draft. The other two authors are advisors: they review the scientific question,
experimental contract, and manuscript, rather than owning implementation tasks.
AI assistance can support implementation and review, but does not replace author
decisions or verification.

This replaces the earlier parallel-author schedule. Historical plans remain at
[archive/docs/PLAN_2026-05.md](archive/docs/PLAN_2026-05.md). Repository cleanup is
complete. The initial scale-contract fix is implemented and CPU-tested; corrected
training and the controlled evaluation are **not** complete. See
[the implementation boundary](docs/CORRECTED_SCALE_CONTRACT.md). The immediate
goal is a defensible publication decision, not another open-ended model search.

**Current resource constraint:** GPU use is prohibited while it is reserved for
the user's urgent work. Only CPU code, tiny fixture checks, and preparatory work
are authorized in this phase. Do not launch training or process-control queues.

## 1. The question and scope limit

> Can we request meaningful trajectory properties and generate diverse,
> dynamically valid Lotka–Volterra-like curves by targeting a learned latent space?

This is the clarified original objective. Matching observation noise and validating
dynamical diagnostics support that objective; they should not replace it with an
evaluation-method paper by default. A visibly sorted latent projection is not a
requirement for useful target-guided generation.

Start with an ID case study and at most three trained configurations: corrected
LSTM-VAE, Transformer-VAE, and the LSTM learned-variance variant (M8). Historical
models are context, not corrected evidence. Do not make architectural superiority,
mode-collapse, or chaos-recovery claims the condition for success.

**New CPU feasibility evidence:** the original sorted v1 checkpoint can target
community overshoot and relative tail level through bank-guided local search.
See [the pilot record](docs/LATENT_CONTROL_PILOT.md). This is not physical GLV
validation, and it does not mandate training all three configurations. First
establish whether the existing model can support the original controlled-generation
claim with appropriate validity/diversity/simulator baselines.

**Default exclusions:** no KAN/latent-ODE retraining, new architecture, latent-space
interpretability campaign, real-world dataset, long-horizon extension, or full OOD
matrix before the core question is answered. Keep old artifacts unchanged.

## 2. First work session: establish the experiment contract

- [ ] Open `docs/SCIENTIFIC_AUDIT.md` and turn each P0 issue below into a testable task.
- [ ] Create a short experiment specification under `configs/` or `docs/` with:
  model/data versions, splits, noise law, time grid, sampling mode, primary metrics,
  post-processing policy, training/evaluation seeds, and a compute cap.
- [ ] Set a personal time budget in actual working days; estimates below assume
  roughly 4–6 focused hours per day, not three parallel implementers.
- [ ] Send the specification and audit to both advisors. Ask for one 30–45 minute
  review and decisions on the claim boundaries, not a general brainstorm.

**Advisor checkpoint A:** agree that noise fidelity and underlying trajectory
fidelity are separate targets. Default to reporting a short-window divergence
statistic, not physical Lyapunov exponents or chaos preservation.

## 3. Correctness gate — approximately 4–6 working days

Work in this order. Add regression tests before changing scientific behavior.
Use new versioned data/checkpoint/output names; do not overwrite historical runs.
Store new experiment metrics and provenance in `results/runs/<run-id>/`.
Historical `RESULTS*` files are grouped in `results/legacy/`; no result files
belong at the repository root.

### A. Correct the unsorted-species scale contract

- [x] Update the selected model families to represent all seven relative scales;
  remove the assumption that species 0 is the dominant species.
- [x] Update the scale loss, reconstruction path, generation path, and per-species
  evaluation consistently. Document how positivity and the family-scale reference
  are represented; do not silently change normalization semantics.
- [x] Version the model/configuration so old six-output checkpoints cannot load as
  corrected models without an explicit legacy path.
- [x] Test every possible dominant-species index, normalization round trips, output
  shapes, all-seven-species loss/scoring, and checkpoint/configuration compatibility.

**Likely files:** `src/models/cvae.py`, `src/models/cvae_heteroscedastic.py`,
`src/models/transformer_vae.py`, `train_cvae.py`, and
`analysis/evaluate_all_models.py`. Unselected historical models need not be
refactored in this phase, but must not be labeled corrected.

**Done when:** synthetic examples with each of the seven dominant indices satisfy
the same scale contract, and the selected models pass a tiny forward/backward/
generation smoke test. A high R² is not the acceptance criterion.

**CPU implementation complete:** explicit `species_v2` versus `legacy_sorted_v1`,
all-species centered log-ratio loss/scoring, versioned corrected checkpoints, and
legacy-unsorted training guards are tested. This changes scale-loss units, so
effective weights still need validation. No old checkpoint was upgraded, no model
was fitted, and CUDA/AMP behavior has not been exercised.

### B. Make the data-generating process explicit and reproducible

The tested single-system foundation now exists in
`data_generation/controlled_glv.py`: physical time, paired clean/noisy outputs,
explicit seeds/selection/failures, and grouped splits. The existing generators
remain historical; integration into a versioned dataset builder is still pending.
`configs/controlled_study_v2.proposed.json` records the proposed contract and open
decisions; it does not schedule data generation or training.

- [ ] Share one simulator implementation/configuration between ID and any future
  parameter-shift experiment. Record exponential **scale**, not ambiguous rate labels.
- [ ] Retain clean trajectories and their noisy observations together, plus
  `(r, A)`, initial conditions, time grid, parameter/noise seeds, and rejection reasons.
- [ ] Make trajectory selection explicit. Prefer selection on clean trajectories
  for the controlled study so changing observation noise does not change which
  systems are accepted; document this departure from the old pipeline.
- [ ] Use separate train, validation, and fresh test splits. Split by underlying
  system before deriving noisy copies, so one system cannot leak across splits.
- [ ] Fix the time horizon, sampling interval, solver settings, and selection rule.
  If OOD is later included, only the declared intervention may change.
- [ ] Remove test-distribution scale borrowing from the primary evaluation. Use a
  documented training-derived scale sampler if needed, and disclose that this is
  an empirical scale component rather than a fully learned absolute-scale model.
- [ ] Test deterministic replay, split separation, clean/noisy pairing, metadata
  completeness, and preprocessing round trips before generating production data.

**Done when:** a small fixture reproduces exactly from its configuration and seeds,
and every accepted trajectory has its physical time and parameter metadata.

### C. Separate generated means, sampled observations, and post-processing

- [ ] Give each selected model explicit mean-output and observation-sampling modes.
  Distinguish observation-only noise from stochastic feedback into future states.
- [ ] Remove or explicitly control M8's mean-feedback training versus sampled-
  feedback generation mismatch. Do not change both feedback and noise law silently.
- [ ] Make unthresholded generation the primary result. Keep irreversible extinction
  thresholding only as a labeled ablation, with its alteration rate reported.
- [ ] Persist evaluation settings, raw per-trajectory diagnostics, and invalid/
  failed-sample counts. Do not hide failure rates behind imputation or NaN exclusion.

**Done when:** a fixed latent draw can be evaluated as a mean trajectory and as
repeated noisy observations without confusing the two estimands.

## 4. Cheap validation and literature gate — approximately 2–3 working days

Do this before a multi-seed training batch. Early metric checks can use simple
signals and small simulator fixtures; they do not need corrected trained models.

- [ ] Persist a diagnostic-control report for constant/relaxing, periodic, IID-noise,
  and clean/noisy simulator signals. Include embedding/threshold sensitivity and
  per-species checks alongside the species average. Constants should be handled
  explicitly as degenerate inputs, not interpreted as meaningful chaos estimates.
- [ ] Estimate real-versus-real reference variability using independent reference
  sets. Compare clean with clean and noisy with noisy at matched sample sizes.
- [ ] Lock effect-size summaries and uncertainty reporting before inspecting the
  final test set. Distinguish trajectory variation from training-seed variation;
  three seeds are a small replication set, not grounds for strong asymptotic claims.
- [ ] Freeze feature scaling/kernel settings from training/validation reference data
  for model comparisons. Use p-values as secondary diagnostics, not a quality score.
- [ ] Spend one focused literature session checking the closest work on scientific
  time-series generation evaluation, observation noise, and sequential VAE likelihoods.
  Produce a small comparison table: prior question, control, result, and our difference.
- [ ] Write provisional Methods and a limitations outline while these details are fresh.

**Advisor checkpoint B:** send a short report containing the controls, closest prior
work, and proposed final comparison. Ask: is there a useful, distinguishable question
left? If not, narrow or stop before paying for a large rerun. A project being old is
not a reason to promise novelty or a particular venue.

## 5. Minimum decisive experiment — pilot first, then replicate

The first CPU-only target-control pilot is complete. The next decisive experiment
should test **requested controls, GLV validity, conditional diversity, and a direct
simulator baseline**. The noise/architecture comparison below is supporting work
only if needed to explain a failure or justify a specific model claim. Do not
automatically launch nine training runs before validating the control result.

### Evaluation variants

| Variant | Purpose | Extra training? |
|---|---|---|
| Corrected LSTM decoder means | Underlying trajectory-distribution baseline | One configuration |
| Same LSTM + validation-calibrated observation noise | Simple practical alternative to a learned variance head | No |
| Same LSTM + simulator-known multiplicative noise | Oracle diagnostic: how much mismatch is observation noise alone? | No |
| Matched LSTM learned-variance decoder | Test whether learning variance adds useful fidelity/calibration | One configuration |
| Corrected Transformer means and matching noise variants | Limited check that the result is not specific to one backbone | One configuration |

Calibrate noise on training/validation data only. Label simulator-known noise as
an oracle, not a deployable learned result. Apply multiplicative observation noise
in original coordinates, or derive and document the equivalent normalized-space
law; it is not interchangeable with constant additive noise.

Comparing MSE and NLL also changes loss scale relative to KL and scale-head terms.
Record those effective weights and control training budgets. If claiming a benefit
specifically from heteroscedasticity, a matched fixed-variance NLL training control
is required. Otherwise drop that architectural claim rather than quietly expanding
the study. Do not call post-hoc noise addition a new architecture.

### Execution order and budget

- [ ] Run one short training smoke test, then one full-seed LSTM pilot. Verify
  checkpoint saving, evaluation, failure accounting, and configuration hashes.
- [ ] Check whether simple calibrated noise already accounts for most of the
  observed discrepancy. This determines the next comparison, not a p-value threshold.
- [ ] Run one seed of each remaining selected configuration only if the pilot and
  literature gate justify it. Save progress checkpoints so interruptions are recoverable.
- [ ] If the comparison is interpretable, run three seeds per retained configuration:
  **at most nine main training runs**, not the full historical architecture matrix.
- [ ] Freeze settings before evaluating the fresh final test set. Report each seed
  and effect sizes, plus appropriately scoped uncertainty, not just a pooled mean.

Reserve approximately 2–4 researcher-days for orchestration, checks, and analysis,
plus measured GPU queue time. Measure the corrected pilot before estimating the
full compute cost; earlier run times are not a guarantee.

**Primary outputs:** a clean-dynamics comparison, an observed-sample comparison,
and a table of reconstruction, structural/distributional error, calibration where
applicable, and invalid-sample rates. Unconditional generated samples are not
pointwise paired reconstructions of test trajectories.

## 6. Publication decision — no automatic expansion

| Controlled outcome | Next action |
|---|---|
| Requested controls are reliable, outputs are dynamically valid/diverse, and simulator baselines establish practical value | Draft the focused controlled-generation paper |
| Requested aggregate properties are attained but individual trajectories fail physical validation | Address validity; do not call it controlled GLV generation yet |
| Bank lookup or direct simulator sampling is sufficient and the learned generator adds no useful benefit | Narrow the contribution honestly rather than inventing latent-order or architectural novelty |
| Noise matching changes the fidelity verdict | Use it to qualify the control result; an evaluation-caution paper is an alternative only if separately justified |
| Effects are not robust or no useful distinction from prior work remains | Discuss a smaller technical report/preprint or stop; do not add models by default |

**Advisor checkpoint C:** one decision meeting with the evidence table and a
proposed abstract. Agree on one claim and scope before another experiment cycle.
Aim to reach this checkpoint after approximately **8–13 focused working days**,
plus advisor response/compute delays. This is a decision target, not a submission
promise or a reason to ignore a failed correctness gate.

OOD is optional after this checkpoint. A fixed unconditional model cannot be
expected to adapt to a growth-rate change it is never conditioned on. Include a
matched OOD experiment only if it tests an explicit claim; do not repeat the old
three-distribution matrix merely because those files already exist.

## 7. Write and release the smallest defensible paper

- [ ] Complete the draft, building on Methods written earlier: question, controlled
  setup, findings, negative controls, limitations, and practical recommendation.
- [ ] Limit the main presentation to four evidence blocks: matched-data design,
  reconstruction versus structural fidelity, noise-control comparison, and residual
  model differences/trade-offs. Keep failed exploratory stories out of the main narrative.
- [ ] Give both advisors the same complete draft, a short claim-to-evidence table,
  and a requested review date. Carlos implements the combined revisions.
- [ ] Budget approximately 5–8 additional focused writing/revision days plus advisor
  turnaround, rather than assuming writing will fit around endless experiments.
- [ ] Select final figures/results and regenerate only those from versioned outputs.
- [ ] Document exact dataset/checkpoint retrieval, license, vendored-code attribution,
  configuration, and reproduction commands before release/submission.

## Immediate next task

**Next CPU task: inspect the targeted outputs per species and establish a
defensible GLV-validity/diversity comparison against simulator-based controls.**
The paired simulator/dataset builder and diagnostic controls support that work.
Keep model fitting and large dataset generation on hold. Send the control-pilot
record, proposed contract, and claim boundaries to the advisors; do not launch a
retraining queue merely to seek prettier latent ordering.
