#!/bin/bash
# Follow-up to post_audit_pipeline.sh.
#
# The original watcher (PID 2332367) is running the old script (file inode
# replaced after edit), so it won't run T1/T2/T3. This script waits for
# the original watcher to exit, then runs:
#   T1 — OOD power-analysis re-eval (N_CHAOS=1000)
#   T2 — Cross-distribution real-stats characterization
#   T3 — Multi-seed noise-addition sweep (17 ckpts)
#
# Args: original watcher PID to wait for.

set -u
cd "$(dirname "$0")/.."
source TimeSeries/bin/activate
mkdir -p logs

WATCHER_PID=${1:?usage: $0 WATCHER_PID}
LOG=logs/post_pipeline_followup.log
exec >> "$LOG" 2>&1

date_stamp() { date -u +'[%Y-%m-%d %H:%M:%S UTC]'; }
log() { echo "$(date_stamp) $*"; }

log "=== follow-up watcher started, waiting for PID=$WATCHER_PID to exit ==="
while kill -0 "$WATCHER_PID" 2>/dev/null; do
  sleep 60
done
log "Original watcher exited."

# Safety: wait for any GPU-using job to settle
log "Waiting 30s for GPU to settle..."
sleep 30
while pgrep -f "train_pivot.py" > /dev/null 2>&1 || pgrep -f "evaluate_all_models.py" > /dev/null 2>&1; do
  log "  GPU job still running, waiting..."
  sleep 60
done
log "GPU is free."

# Build the same ckpt list the main pipeline uses
declare -A TYPE
TYPE[1]=cvae-scale-cond
TYPE[2]=cvae-no-scale-cond
TYPE[3]=cvae-stochastic
TYPE[4]=latent-ode
TYPE[5]=transformer-vae
TYPE[6]=kan-vae

build_ckpts() {
  local ckpts=()
  for m in 1 2 3 4 5 6; do
    for seed in 42 123 2026; do
      local p=model_ckpts/model_${m}_seed${seed}.pth
      if [[ -f "$p" ]]; then
        ckpts+=("${TYPE[$m]}=$p")
      fi
    done
  done
  echo "${ckpts[@]}"
}

# ----- T1: OOD power-analysis re-eval (N_CHAOS=1000 instead of 200) -----
log "=== T1: OOD power-analysis (N_CHAOS=1000) ==="
for tset in TEST_FINAL_NOSORT TEST_OOD_Exp1 TEST_OOD_Exp5; do
  out_suffix="${tset#TEST_}"
  out_path="RESULTS_COMPARATIVE_MULTISEED_N1000_${out_suffix}.json"
  log "Eval N_CHAOS=1000 on $tset → $out_path"
  python analysis/evaluate_all_models.py \
    --checkpoints $(build_ckpts) \
    --test-path "data/${tset}.pkl" \
    --out "$out_path" --force \
    --n-chaos 1000 2>&1 || log "  (eval failed, continuing)"
done
log "=== T1 done ==="

# ----- T2: Cross-distribution real-stats characterization -----
log "=== T2: cross-distribution real-stats ==="
python -u <<'PYEOF' || log "T2 failed"
import json
import pickle
import sys
sys.path.insert(0, '.')
import numpy as np
from analysis.chaos_diagnostics import rqa_measures, rosenstein_lyapunov

out = {}
for tset in ['TEST_FINAL_NOSORT', 'TEST_OOD_Exp1', 'TEST_OOD_Exp5']:
    with open(f'data/{tset}.pkl', 'rb') as f: pkg = pickle.load(f)
    rng = np.random.default_rng(20260520)
    N = pkg['data'].shape[0]
    idx = rng.choice(N, size=min(500, N), replace=False)
    real = (pkg['data'][idx]
            * pkg['reconstruction_max_values'][idx][:, :, None]
            * pkg['family_max_values'][idx][:, None, None])
    dets, lyaps = [], []
    for t in real:
        sig = t.mean(axis=0)
        dets.append(rqa_measures(sig)['DET'])
        lyaps.append(rosenstein_lyapunov(sig))
    dets, lyaps = np.array(dets), np.array(lyaps)
    out[tset] = {
        'n': int(len(dets)),
        'DET_mean': float(dets.mean()), 'DET_std': float(dets.std()), 'DET_median': float(np.median(dets)),
        'lyap_mean': float(np.nanmean(lyaps)), 'lyap_std': float(np.nanstd(lyaps)), 'lyap_median': float(np.nanmedian(lyaps)),
    }
    print(f'{tset}: DET={dets.mean():.4f}±{dets.std():.4f}, lyap={np.nanmean(lyaps):+.4f}+/-{np.nanstd(lyaps):.4f}')

with open('RESULTS_REAL_STATS_CROSSDIST.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Wrote RESULTS_REAL_STATS_CROSSDIST.json')
PYEOF
log "=== T2 done ==="

# ----- T3: Multi-seed noise-addition sweep (17 ckpts) -----
log "=== T3: noise-addition sweep, all 17 ckpts ==="
python analysis/noise_addition_sweep.py --out RESULTS_NOISE_SWEEP_MULTISEED.json 2>&1 || log "T3 failed"
log "=== T3 done ==="

log "=== follow-up FINISHED ==="
