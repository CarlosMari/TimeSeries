# Historical result files

This folder contains the 40 JSON metric files and two generated Markdown summaries
previously at the repository root. Their contents were moved without recomputing,
reserializing, or normalizing whitespace. Existing local changes were retained.

These are legacy experiments, not the corrected study in `PLAN.md`. See the
[result-family index](../README.md) and
[scientific audit](../../docs/SCIENTIFIC_AUDIT.md) before interpreting them.

Active legacy analysis scripts and queue commands now read/write paths here.
They can still overwrite historical metrics when explicitly run. Prefer a unique
`results/runs/<run-id>/` output when evaluating a new configuration. Do not execute
the process-control queues simply to check path migration.

No root-level compatibility copies or symlinks are kept. If a private notebook
uses `RESULTS_X.json` relative to the repository root, update it to
`results/legacy/RESULTS_X.json`. Archived guides retain the names they originally
used and are not current reproduction instructions.
