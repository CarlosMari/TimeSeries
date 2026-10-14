# Results index — no designated final experiment

All 40 existing `RESULTS*.json` files and both generated Markdown summaries are
grouped under [`legacy/`](legacy/README.md). The relocation preserves their exact
bytes, including existing local modifications. Active analysis scripts and shell
queues use the new paths; archived narratives retain their historical references.
Moving results does not validate or endorse them.

## Result families

| Filename family in `legacy/` | What it represents | Current status |
|---|---|---|
| `RESULTS.json`, `RESULTS_BASELINE.json`, `RESULTS_NOVELTY.json`, `RESULTS_CHAOS.json`, `RESULTS_PARAM_RECOVERY.json` | Original single-model analyses | Historical with-sort study; not interchangeable with no-sort results |
| `RESULTS_COMPARATIVE*.json` | Comparative evaluations, including partial, single-seed, multiseed, and N1000 variants | Filename variants are not a verified provenance chain; no-sort scale-head issue remains |
| `RESULTS_B1*.json` | Frozen decoder-noise and spectral-loss interventions | Exploratory controls, not validated cures |
| `RESULTS_NOISE_SWEEP*.json` | Post-hoc observation-noise sweeps | Mean/sample distinction and noise selection need controlled evaluation |
| `RESULTS_M8*.json` | Learned variance-head experiment | Different loss and observation sampling; not isolated heteroscedastic benefit |
| `RESULTS_LENS_VALIDATION.json` | Synthetic perturbation sensitivity checks | Not proof of chaos-estimation validity or calibrated specificity |
| `RESULTS_REAL_STATS_CROSSDIST.json` | Dataset diagnostic summaries | Current ID/OOD generator differences are compound |

Any family containing `OOD` inherits the ID/OOD noise, time-horizon, and filtering
confounds. A filename containing `FINAL` does not supersede those caveats.

[`legacy/RESULTS.md`](legacy/RESULTS.md) is a historical generated single-model
summary; its old "verified" label is not the status of the comparative paper.
[`legacy/RESULTS_COMPARISON.md`](legacy/RESULTS_COMPARISON.md) is the corresponding
historical baseline comparison. Root `REFERENCES.md` is a citation working list,
not a validated bibliography.

## Artifact catalog

```bash
python3 scripts/catalog_artifacts.py --output results/manifest.json
python3 scripts/catalog_artifacts.py --output results/manifest.json --check
```

The manifest inventories available metrics, local data files, and checkpoints,
with byte counts, checksums where requested, discoverable metadata, and explicitly
unverified inferences. Large data/checkpoint files are not loaded or deserialized.
The manifest does not prove which exact bytes were used to produce an old result.
It describes the local working tree, including untracked artifacts and uncommitted
contents, not just a Git commit. A fresh clone without those local files will have
a different inventory; regenerate it locally rather than interpreting a stale
catalog check as corrupted published evidence.

Future paper runs should record code revision, full configuration, train/validation/
test identities, checkpoint hashes, evaluation RNG seeds, preprocessing, post-
processing, diagnostic settings, failure rates, and environment alongside outputs.
Use `results/runs/<run-id>/` for new experiments and explicit `--out` paths rather
than overwriting historical metrics. Generated run directories remain ignored by
default; select versioned release evidence deliberately. The current catalog is
scoped to historical metrics plus local datasets/checkpoints, not arbitrary new
run outputs.
Choose the canonical paper subset only after the correctness gate in `PLAN.md`.
