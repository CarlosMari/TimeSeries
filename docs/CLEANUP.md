# Repository cleanup record

## Scope

The October 2026 cleanup organizes the repository and adds maintenance tooling.
It does **not** modify model math, RNG behavior, data generation, training loops,
scientific metric calculations, saved result contents, or publication figures. The scientific fixes in
[SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md) remain separate work.

## Relocations

- Previous root `PROJECT.md` and `PLAN.md` are preserved as
  `archive/docs/PROJECT_2026-05.md` and `archive/docs/PLAN_2026-05.md`.
- Historical root guides, model comparisons, interpretation summaries, figure
  inventories, and the threshold report moved under `archive/docs/`, retaining
  their original basenames.
- Earlier notes formerly directly under `docs/` moved to `archive/docs/notes/`.
  The May design specification remains at its previous path, marked historical.
- Root exploratory Python scripts moved to `archive/scripts/`, retaining their
  basenames and original source contents. The four `train_*.py` entry points stay.
- The two old root figure shell wrappers moved with those scripts as historical
  records, not supported launchers. `scripts/run_legacy.py` provides a separate,
  opt-in execution path for historical Python scripts.
- Root LaTeX figure/caption sources, associated root architecture PDFs, and the data-generation manuscript fragment
  moved to `archive/paper/`. These are fragments, not a complete manuscript build.
- The original broad `requirements.txt` is preserved as
  `archive/environment/requirements-legacy.txt`; direct numerical dependencies and
  optional exploration dependencies now have separate root requirement files.

Historical sources retain their original claims and relative references, so links
inside archived documents may need to be interpreted relative to their old location.
Current instructions and interpretation live outside the archive.

## Deliberately preserved in place

- Datasets, model checkpoints, figure directories, local virtual environments,
  logs, W&B runs, and existing ignored experimental directories.
- The essential pre-existing `src/utils/` files were included in the first cleanup
  commit; cleanup never treats untracked configuration as disposable.
- The pre-existing deletion of `checkpoints/TRAIN_FINAL_checkpoint.pkl` is not
  restored, and pre-existing modified figures are not regenerated.

## Follow-up: consolidate results after explicit relocation approval

The first cleanup retained root result paths for compatibility. The user then
explicitly requested moving them. All 40 root JSON metric files and both generated
Markdown summaries now live under `results/legacy/`, with byte identity checked
against a pre-move inventory. This includes pre-existing modified and untracked
results; no metrics were recomputed or reserialized.

Active Python readers/writers, CLI defaults, comparative-figure fallback paths,
shell queue arguments, and embedded shell/Python writers use the new directory.
Custom CLI output directories are created explicitly. Archived sources remain
unchanged, so their old relative result references are historical, not executable
instructions. No root compatibility files or symlinks remain.

The old `results/` directory also contains ignored historical exploratory material.
Only its index/catalog and the named legacy metrics/summaries are exposed to version
control; cleanup does not blanket-unignore unrelated outputs. New corrected runs
should use `results/runs/<run-id>/` with explicit provenance.

The pre-existing ignored `results/README.md` was replaced by the expanded current
result index. Unlike the relocated historical sources, that short directory guide
was not archived. It is an intentional documentation replacement, not a preserved
scientific artifact; the original pre-cleanup hash and size were recorded in the
external inventory.

## Validation boundary

Before changes, an external inventory recorded hashes and git status. Relocated
source files are checked against that inventory; data, checkpoint, result, and
figure bytes must remain unchanged. Maintenance tests use temporary fixtures and
do not load scientific artifacts or run the historical shell queues.

`archive/manifest.json` records each relocation's original and archived paths,
original size, and original SHA256. `python3 scripts/check_archive.py` verifies
that the preserved archive still matches those pre-cleanup bytes without running
any historical code. Replacement root documentation is intentionally not compared
with the old narratives it supersedes.

The first cleanup was committed and pushed with user authorization. Result-path
migration is a separate follow-up. Neither phase deletes large data, converts
checkpoints, launches training, or recomputes scientific metrics. Selecting final
publication evidence remains separate from organizing the legacy results.
