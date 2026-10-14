# Historical material — preserved, not endorsed

This directory separates previous project narratives and exploratory code from
the maintained entry points. Archived source contents are preserved; relocation
does not validate their claims, repair their dependencies, or select their figures
for the paper. Relative Markdown and LaTeX paths inside historical documents may
refer to their original repository-root location.

| Directory | Contents |
|---|---|
| `docs/` | Previous root guides, result interpretations, and May 2026 project/plan snapshots |
| `docs/notes/` | Earlier notes formerly under `docs/`, including obsolete agent guidance |
| `scripts/` | Root exploratory analysis and figure scripts, plus historical shell wrappers |
| `paper/` | Historical LaTeX figure/caption sources and manuscript fragments |
| `environment/` | Original broad requirements snapshot, not the supported installer |

`manifest.json` maps relocated files to their original locations and byte hashes.
Verify preservation with `python3 scripts/check_archive.py` from any working
directory when using an absolute path to that command.

## Running a historical Python script

List scripts without importing or executing them:

```bash
python3 scripts/run_legacy.py --list
```

To opt in to executing one with the selected Python environment:

```bash
TimeSeries/bin/python scripts/run_legacy.py --execute analyze_failure_cases.py
```

The launcher uses the repository root as the working directory and adds both the
root and historical script directory to Python's import path. This preserves
root-relative data/checkpoint/output paths and sibling imports without injecting
code into the historical sources. Arbitrary file paths and shell scripts are not
accepted by this launcher.

**Execution can overwrite figures, write artifacts, require CUDA, or fail because
local inputs are absent.** Some scripts use illustrative/synthetic histories or
demonstrations; executing them is not evidence of a scientific reproduction.

The archived `generate_all_paper_figures.sh` assumes the former root script layout;
it is retained as a record, not a supported launcher. `regenerate_all_30d.sh` also
references missing `quick_*30d.py` helpers. Do not run either as a reproduction
command. Historical LaTeX fragments are likewise not a complete manuscript build.
