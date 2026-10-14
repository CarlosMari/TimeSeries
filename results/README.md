# Results index — no designated final experiment

Existing metrics remain in root `RESULTS*.json` files because historical analysis
scripts and shell queues read/write those exact paths. This directory adds a
catalog; it does not relocate, rewrite, or endorse the metrics.

## Result families

| Root filename family | What it represents | Current status |
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

`RESULTS.md` is a historical generated single-model summary; its old "verified"
label is not the status of the comparative paper. `REFERENCES.md` is a citation
working list, not a validated bibliography.

## Artifact catalog

```bash
python3 scripts/catalog_artifacts.py --output results/manifest.json
python3 scripts/catalog_artifacts.py --output results/manifest.json --check
```

The manifest inventories available metrics, local data files, and checkpoints,
with byte counts, checksums where requested, discoverable metadata, and explicitly
unverified inferences. Large data/checkpoint files are not loaded or deserialized.
The manifest does not prove which exact bytes were used to produce an old result.

Future paper runs should record code revision, full configuration, train/validation/
test identities, checkpoint hashes, evaluation RNG seeds, preprocessing, post-
processing, diagnostic settings, failure rates, and environment alongside outputs.
Choose the canonical paper subset only after the correctness gate in `PLAN.md`.
