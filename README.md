# Generative Models of Generalized Lotka–Volterra Time Series

Research code for reconstructing and generating 7-species, 65-step ecological
trajectories. The repository contains LSTM, latent-ODE, Transformer, and KAN VAE
variants, an experimental GLV-regression baseline, and distributional diagnostics.

**Status: scientific audit and repository cleanup, October 2026.** Existing
results are research artifacts, not a validated final benchmark. The no-sort
scale head, compound OOD shift, observation-noise comparisons, and interpretation
of short-window divergence estimates need resolution before publication claims.
Cleanup does not change model behavior or certify old results.

## Start here

- [PROJECT.md](PROJECT.md): current scope and evidence status.
- [PLAN.md](PLAN.md): bounded next steps and publication decision gate.
- [docs/SCIENTIFIC_AUDIT.md](docs/SCIENTIFIC_AUDIT.md): concrete scientific blockers.
- [docs/REPRODUCING.md](docs/REPRODUCING.md): environment, safe checks, and explicit
  historical reproduction commands.
- [docs/CLEANUP.md](docs/CLEANUP.md): what moved, what was preserved, and why.
- [results/README.md](results/README.md): result families and provenance limitations.

## Safe checks (no training or artifact regeneration)

These use only the Python standard library and run without CUDA or local data:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/check_archive.py
python3 scripts/run_legacy.py --list
python3 scripts/catalog_artifacts.py --output results/manifest.json
python3 scripts/catalog_artifacts.py --output results/manifest.json --check
```

The catalog command inventories existing artifacts; it does not load pickle files,
load checkpoints, run models, or recompute scientific metrics. Paths and filenames
can suggest provenance but cannot prove it.

## Layout

| Path | Purpose |
|---|---|
| `train_*.py` | Existing training entry points; CUDA required by current configuration |
| `src/models/`, `src/utils/` | Model implementations and shared configuration |
| `data_generation/` | Simulator and preprocessing; ID/OOD differences remain under audit |
| `analysis/` | Existing evaluation and comparative figure code |
| `scripts/` | Maintenance tools and historical training queues |
| `tests/` | Lightweight automated tests plus separately named legacy checks |
| `results/` | Artifact catalog and result-family documentation |
| `results/legacy/` | Historical metrics and generated summaries; active legacy scripts use these paths |
| `data/`, `model_ckpts/` | Local datasets and checkpoints; not supplied by a clean clone |
| `final figures/`, other figure directories | Preserved historical outputs, not a curated final paper |
| `archive/` | Preserved historical documentation, exploratory scripts, and paper fragments |

New corrected experiments should use a unique `results/runs/<run-id>/` directory
with configuration and provenance, not overwrite legacy metrics.

Historical exploratory scripts run through `scripts/run_legacy.py`; see
[archive/README.md](archive/README.md) before executing anything. They may overwrite
old outputs, use synthetic demonstrations, or depend on missing local artifacts.

## Environment and licensing

Use `requirements.txt` for the numerical/training pipeline and
`requirements-exploration.txt` for optional exploratory tools. The original broad
environment snapshot is preserved under `archive/environment/`.

A distribution license has not yet been selected. Resolve licensing, vendored-code
attribution, and dataset/checkpoint availability before a public research release.
