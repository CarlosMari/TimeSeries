#!/bin/bash
# Post-audit pipeline.
#
# Triggered when seed-2026 m5 finishes (the last seed-2026 VAE-family training
# in the autoqueue). Skips the queued m7_seed2026 because m7 was dropped from
# the paper per audit §1.3.3.1.
#
# Tier 1 work after m5_seed2026 lands:
#   1.1  multi-seed unified eval (all 6 architectures × 3 seeds × 3 distributions)
#   1.2  noise-addition sweep figure
#   1.3  B1 σ=0.05 cure multi-seed verification (train m1_stoch frozen 0.05 on
#        seeds 123 + 2026, eval)
#   1.4  regenerate comparative figures from multi-seed JSON
#
# Order rationale:
#   - 1.2 needs only ckpts on disk → can run RIGHT NOW in parallel with autoqueue
#     (CPU-bound, no contention) — handled separately, not here.
#   - 1.1 needs all seed-2026 VAE-family ckpts (m1, m2, m3, m4, m5)
#   - 1.3 needs the GPU — run AFTER 1.1
#   - 1.4 needs RESULTS_COMPARATIVE.json populated
#
# Logs to logs/post_audit_watcher.log

set -u
cd "$(dirname "$0")/.."
source TimeSeries/bin/activate
mkdir -p logs

AUTOQUEUE_PID=${1:?usage: $0 AUTOQUEUE_PID}

LOG=logs/post_audit_watcher.log
exec >> "$LOG" 2>&1

date_stamp() { date -u +'[%Y-%m-%d %H:%M:%S UTC]'; }
log() { echo "$(date_stamp) $*"; }

log "=== post-audit watcher started, autoqueue PID=$AUTOQUEUE_PID ==="

if ! kill -0 "$AUTOQUEUE_PID" 2>/dev/null; then
  log "FATAL: autoqueue PID $AUTOQUEUE_PID does not exist"
  exit 1
fi

# Block until m5_seed2026 lands.
log "Waiting for model_5_seed2026.pth to land..."
while [[ ! -f model_ckpts/model_5_seed2026.pth ]]; do
  sleep 60
done
log "model_5_seed2026.pth detected."

# Autoqueue will spawn m7_seed2026 next. SIGSTOP autoqueue immediately, then
# kill any m7_seed2026 process the autoqueue may have already launched.
log "Pausing autoqueue (SIGSTOP $AUTOQUEUE_PID) to prevent m7_seed2026 spawn..."
kill -STOP "$AUTOQUEUE_PID" || { log "FATAL: SIGSTOP failed"; exit 1; }
sleep 5
STATE=$(ps -o stat= -p "$AUTOQUEUE_PID" | tr -d ' ')
log "Autoqueue state: $STATE"

# Kill m7 if it's already running (autoqueue may have spawned it in the gap)
if pgrep -f "model_7_seed2026" > /dev/null; then
  log "m7_seed2026 was already spawned — killing it (dropped from paper)"
  pkill -f "model_7_seed2026" || true
  sleep 5
fi

# Wait for any other GPU-using training to exit cleanly
log "Waiting for any GPU job to release..."
while pgrep -f "train_pivot.py" > /dev/null 2>&1 || pgrep -f "train_glv_regression.py" > /dev/null 2>&1; do
  sleep 30
done
log "GPU is free."
sleep 30  # safety pause for GPU memory release

# Build the ckpt list for Tier 1.1 — all 6 architectures × 3 seeds, m7 excluded
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

# ----- Tier 1.1: multi-seed unified eval -----
log "=== Tier 1.1: multi-seed unified eval ==="

for tset in TEST_FINAL_NOSORT TEST_OOD_Exp1 TEST_OOD_Exp5; do
  out_suffix="${tset#TEST_}"
  out_path="RESULTS_COMPARATIVE_MULTISEED_${out_suffix}.json"
  log "Eval on $tset → $out_path"
  python analysis/evaluate_all_models.py \
    --checkpoints $(build_ckpts) \
    --test-path "data/${tset}.pkl" \
    --out "$out_path" --force 2>&1
done
log "=== Tier 1.1 done ==="

# ----- Tier 1.3: B1 σ=0.05 multi-seed cure verification -----
log "=== Tier 1.3: B1 σ=0.05 cure across seeds 123 + 2026 ==="
for seed in 123 2026; do
  name=b1_m3_frozen_0p05_seed${seed}
  ckpt=model_ckpts/${name}.pth
  if [[ -f "$ckpt" ]]; then
    log "SKIP $name (ckpt exists)"
    continue
  fi
  log "TRAIN $name"
  python train_pivot.py --model cvae-stochastic --seed $seed \
    --name $name --epochs 500 --decoder-noise-init 0.05 --decoder-noise-freeze \
    --wandb-project Conditional_LV_VAE_pivot_b1 > "logs/${name}.log" 2>&1
  rc=$?
  log "DONE $name rc=$rc"

  # Eval on all 3 distributions
  for tset in TEST_FINAL_NOSORT TEST_OOD_Exp1 TEST_OOD_Exp5; do
    out_suffix="${tset#TEST_}"
    out="RESULTS_B1_frozen_0p05_seed${seed}_${out_suffix}.json"
    log "Eval $name on $tset → $out"
    python analysis/evaluate_all_models.py \
      --checkpoints cvae-stochastic=$ckpt \
      --test-path "data/${tset}.pkl" \
      --out "$out" --force 2>&1
  done
done
log "=== Tier 1.3 done ==="

# ----- Tier 1.4: regenerate comparative figures -----
log "=== Tier 1.4: regenerate comparative figures ==="
# Pick one of the multi-seed jsons as the main input
if [[ -f RESULTS_COMPARATIVE_MULTISEED_FINAL_NOSORT.json ]]; then
  python analysis/make_comparative_figures.py --results RESULTS_COMPARATIVE_MULTISEED_FINAL_NOSORT.json 2>&1
fi

# Resume autoqueue (it has nothing left to do, but cleanliness)
log "Resuming autoqueue (SIGCONT $AUTOQUEUE_PID)..."
kill -CONT "$AUTOQUEUE_PID" || log "WARNING: SIGCONT failed"

log "=== post-audit pipeline FINISHED ==="
