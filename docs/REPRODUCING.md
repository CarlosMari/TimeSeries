# Reproduction and environment boundaries

Run commands from the repository root unless stated otherwise. The source tree
does not distribute the full training/test datasets or checkpoint collection.
Only load local pickle/checkpoint files you trust.

## Safe maintenance checks

Python 3.12 was available during cleanup. Maintenance tools use the standard
library. Optional scientific tests use tiny CPU fixtures if numerical dependencies
are installed, and otherwise skip explicitly. No GPU or production data is needed:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/check_archive.py
python3 scripts/run_legacy.py --list
python3 scripts/catalog_artifacts.py --output results/manifest.json
python3 scripts/catalog_artifacts.py --output results/manifest.json --check
```

For the full numerical checks with CUDA hidden, one numerical thread per library,
and W&B disabled, use:

```bash
TimeSeries/bin/python scripts/run_cpu_checks.py
```

**The GPU is currently reserved for urgent work.** Do not run the historical
evaluation/training commands below during this CPU-only phase. They are reference
instructions for later, not scheduled work. See
[the corrected implementation note](CORRECTED_SCALE_CONTRACT.md) for what is tested
and what remains unresolved.

`--check` verifies whether the catalog is current; it does not certify scientific
provenance. Regenerate the catalog when metrics, datasets, or checkpoints change.
Large artifacts are inventoried without deserialization; use the catalog's
explicit large-file hashing option when stronger byte identity is required.

## Numerical environment

To use an existing local environment:

```bash
source TimeSeries/bin/activate
```

For a new environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose a PyTorch wheel/index compatible with the local NVIDIA driver. Dependency
pins describe direct dependencies from the inspected environment, not a tested
cross-platform lock. Optional interactive/legacy plotting dependencies are in
`requirements-exploration.txt`; the previous all-purpose snapshot is preserved in
`archive/environment/requirements-legacy.txt`.

The older `data_generation/glv_functions.py` utilities and notebooks that import
them require `lemkelcp` from the exploration requirements. It is not needed by the
current `custom_glv_FIXED.py` simulator, preprocessing, or comparative evaluator.

Current training configuration sets `DEVICE='cuda'` in `src/utils/config.py`.
Cleanup does not add CPU fallback or change that configuration. Plotting in a
headless session may need `MPLBACKEND=Agg`.

## Historical comparative evaluation — opt-in, writes outputs

The following command evaluates an **existing, scientifically confounded**
checkpoint. It is not the corrected experiment from `PLAN.md`. Supply the local
dataset/checkpoint first. Use a new output filename to protect archived metrics:

```bash
RUN_DIR="results/runs/reproduction-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
python analysis/evaluate_all_models.py \
  --checkpoints cvae-scale-cond=model_ckpts/model_1_seed42.pth \
  --test-path data/TEST_FINAL_NOSORT.pkl \
  --n-chaos 1000 \
  --out "$RUN_DIR/comparative.json"
```

The evaluator reuses existing records unless `--force` is passed. A filename alone
does not encode settings, checkpoint identity, or code version: use a fresh output
for a different configuration and record those inputs separately. Defaults are
2,000 generated samples and up to 5,000 reconstruction samples; this is not a tiny
CPU smoke test.

## Historical preprocessing and training — opt-in, expensive

```bash
python -m data_generation.preprocessor \
  --input data/TRAIN_FINAL_FIXED.pkl \
  --output data/TRAIN_REPRODUCTION_NOSORT.pkl --no-sort
```

Training interfaces are `train_pivot.py`, `train_cvae.py`, `train_baseline.py`, and
`train_glv_regression.py`. The pivot entry point accepts explicit model, seed,
dataset routes, checkpoint name, and epoch count; use `python train_pivot.py --help`
to inspect it. Training may write checkpoints and contact W&B unless logging is
disabled. The pivot CLI now requires explicit `--scale-contract`; use `species_v2`
only for the three selected corrected model families. Legacy training on unsorted
data additionally requires an explicit historical-reproduction override. Old
checkpoints cannot be silently upgraded, and existing checkpoint paths are refused
by the pivot CLI. Fix the remaining scientific contracts before scheduling runs.

Do not execute the shell queues merely to test cleanup. Some discover, stop, resume,
or kill matching processes, and `scripts/autoqueue.sh` assumes writable storage at
`/mnt/storage/shared/TimeSeries`. Their process-control code remains historical;
their training commands predate the new required scale flags and are not an
approved corrected-training launcher. Do not add overrides just to make them run.

## Historical analyses and figures

Many `analysis/` commands use hardcoded output paths and overwrite
`results/legacy/RESULTS*.json`, Markdown summaries in that directory, or files in
`final figures/`. Active consumers and shell queues were updated when the results
moved; root-level `RESULTS*` paths are no longer supported. The historical figure
plotter is not a curated final-paper build and does not establish scientific
validity. Do not regenerate figures as an installation check.

The old root exploration scripts moved to `archive/scripts/`. Use the explicit
launcher documented in [archive/README.md](../archive/README.md), rather than
assuming their former root filenames still exist.
