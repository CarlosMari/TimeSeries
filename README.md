# Comparative Evaluation of Generative Models of GLV Dynamics under a Nonlinear-Dynamics Protocol

A research codebase for the paper **"A three-lens nonlinear-dynamics evaluation protocol for generative models, applied to seven architectures on Generalized Lotka–Volterra trajectories"** (working title). Target venue: *Chaos, Solitons & Fractals* or similar.

## What this is

We compare seven generative-model architectures on a controlled testbed (GLV time series, 7 species, 65 timesteps) under a three-lens evaluation protocol designed to detect dynamical-fidelity defects that recon-quality metrics miss:

1. **Feature-space distributional fidelity** — MMD permutation test on dynamical features.
2. **Recurrence Quantification Analysis** — DET, L_mean, L_max, LAM, TT.
3. **Largest Lyapunov exponent** (Rosenstein) — chaos-sensitivity test.

The seven models span recurrent, attention, continuous-time-ODE, novel-function-basis, and physics-informed inverse-problem inductive biases.

## Status (live)

This project is mid-pivot. As of **2026-05-15**, the codebase contains:

- A working scale-conditioned LSTM-VAE pipeline (the "v1" model).
- All v1 evaluation infrastructure (`analysis/produce_paper_metrics.py`, `analysis/novelty_coverage.py`, `analysis/chaos_diagnostics.py`).
- The pivot design doc and the new pipeline scaffolding.

The 7-model comparative training and unified eval harness are in progress. Track the roadmap in [`PROJECT.md`](PROJECT.md) §1.2 and [`PLAN.md`](PLAN.md).

## Repository map

- [`docs/superpowers/specs/2026-05-15-comparative-evaluation-design.md`](docs/superpowers/specs/2026-05-15-comparative-evaluation-design.md) — **the canonical pivot design doc.** Start here for context.
- [`PROJECT.md`](PROJECT.md) — wiki of current state, v1 results, and pivot roadmap.
- [`PLAN.md`](PLAN.md) — todo list / phase tracker.
- [`REFERENCES.md`](REFERENCES.md) — living citations list grouped by paper section.
- [`src/models/`](src/models/) — model implementations (currently: scale-conditioned LSTM-VAE; the other 6 will land here during phases A–B).
- [`analysis/`](analysis/) — evaluation scripts. The protocol's three lenses live in `novelty_coverage.py` (MMD) and `chaos_diagnostics.py` (RQA + Lyapunov).
- [`data_generation/`](data_generation/) — GLV simulator + preprocessing pipeline.
- [`final figures/`](final%20figures/) — figures for the paper (v1 figures stay archival; pivot figures replace them as they land).

## Reproducing v1 results

```bash
source TimeSeries/bin/activate
python train_cvae.py                              # canonical v1 VAE training
python -c "from analysis.produce_paper_metrics import main; main()"
python analysis/novelty_coverage.py               # Lens 1
python analysis/chaos_diagnostics.py              # Lenses 2 + 3
python analysis/evaluate_baseline.py              # v1 ablation
python analysis/parameter_recoverability.py       # v1 parameter-recovery analysis
```

v1 outputs live in `RESULTS.json`, `RESULTS_BASELINE.json`, `RESULTS_NOVELTY.json`, `RESULTS_CHAOS.json`, `RESULTS_PARAM_RECOVERY.json`.

Reproducing pivot results requires the rebuilt no-sort preprocessing and the 21 new checkpoints — instructions to follow once Phases A–C complete.

## Acknowledgments


## License

(TBD — to be set before submission.)
