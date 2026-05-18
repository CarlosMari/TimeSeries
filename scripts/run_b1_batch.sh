#!/bin/bash
# B1 high-priority experimental batch:
#   - m3 stochastic-decoder with σ FROZEN at 0.05, 0.1, 0.2 (3 runs).
#   - m1-style scale-cond VAE with spectral-loss weight 0.1 (1 run).
#
# These directly test whether the OOD-confirmed "MSE-attractor at DET ≈ 0.99"
# can be broken by either (a) forcing decoder stochasticity, or (b) adding a
# spectral-mismatch penalty to the loss.
#
# Per user direction 2026-05-17 ("ASAP, but don't disturb"), this script waits
# for the autoqueue's *current* child to finish (m5_seed123), then SIGSTOPs
# the autoqueue parent so m6_seed123 doesn't start, runs the B1 batch in
# the freed GPU window, then SIGCONTs the autoqueue.
#
# Logs to logs/b1_watcher.log + logs/b1_<variant>.log per training.

set -u
cd "$(dirname "$0")/.."
source TimeSeries/bin/activate
mkdir -p logs model_ckpts

AUTOQUEUE_PID=${1:?usage: $0 AUTOQUEUE_PID}

LOG=logs/b1_watcher.log
exec >> "$LOG" 2>&1

date_stamp() { date -u +'[%Y-%m-%d %H:%M:%S UTC]'; }
log() { echo "$(date_stamp) $*"; }

log "=== B1 watcher started, autoqueue PID=$AUTOQUEUE_PID ==="

# Sanity: does the autoqueue PID exist?
if ! kill -0 "$AUTOQUEUE_PID" 2>/dev/null; then
  log "FATAL: autoqueue PID $AUTOQUEUE_PID does not exist"
  exit 1
fi

# Step 1: SIGSTOP the autoqueue FIRST, before waiting on its current child.
# SIGSTOP on the parent does NOT propagate to its python child (verified
# 2026-05-17), so the running training continues. But now when it exits, the
# autoqueue can't fire the next item — so when pgrep stops matching, the GPU
# is genuinely ours to take.
#
# (The previous version polled pgrep with a 30s loop while leaving the
# autoqueue running. The autoqueue spawned the next child within
# sub-seconds of the previous one exiting — well within one poll period — so
# the loop never saw a "no children" instant and never proceeded. Race lost.
# Pausing first eliminates the race.)
log "Pausing autoqueue first (SIGSTOP $AUTOQUEUE_PID) so its next item can't fire mid-poll..."
kill -STOP "$AUTOQUEUE_PID" || { log "FATAL: SIGSTOP failed"; exit 1; }
sleep 2
STATE=$(ps -o stat= -p "$AUTOQUEUE_PID" | tr -d ' ')
if [[ "$STATE" != T* ]]; then
  log "WARNING: autoqueue state is '$STATE' (expected T)"
fi

log "Waiting for the autoqueue's current child (if any) to finish..."
while pgrep -f "train_pivot.py" > /dev/null 2>&1 || pgrep -f "train_glv_regression.py" > /dev/null 2>&1; do
  sleep 30
done
log "GPU job ended."

# Verify autoqueue is stopped (otherwise it'll race with us)
sleep 3
STATE=$(ps -o stat= -p "$AUTOQUEUE_PID" | tr -d ' ')
if [[ "$STATE" != T* ]]; then
  log "WARNING: autoqueue state is '$STATE' (expected T); continuing anyway"
fi
log "Autoqueue paused (state=$STATE). Beginning B1 training batch."

# Brief safety wait: 30s for any GPU memory to release
sleep 30

run_b1() {
  local label=$1; shift
  local ckpt=$1; shift
  if [[ -f "$ckpt" ]]; then
    log "SKIP $label (already exists)"
    return
  fi
  log "START $label  $*"
  "$@" > "logs/${label}.log" 2>&1
  local rc=$?
  log "DONE  $label  rc=$rc"
}

# Three frozen-σ variants of the stochastic-decoder VAE (model 3 family).
# Same training schedule as model 3 seed 42: 500 epochs, scale-conditioning on.
run_b1 b1_m3_frozen_0p05_seed42  model_ckpts/b1_m3_frozen_0p05_seed42.pth \
  python train_pivot.py --model cvae-stochastic --seed 42 \
    --name b1_m3_frozen_0p05_seed42 --epochs 500 \
    --decoder-noise-init 0.05 --decoder-noise-freeze \
    --wandb-project Conditional_LV_VAE_pivot_b1

run_b1 b1_m3_frozen_0p1_seed42  model_ckpts/b1_m3_frozen_0p1_seed42.pth \
  python train_pivot.py --model cvae-stochastic --seed 42 \
    --name b1_m3_frozen_0p1_seed42 --epochs 500 \
    --decoder-noise-init 0.1 --decoder-noise-freeze \
    --wandb-project Conditional_LV_VAE_pivot_b1

run_b1 b1_m3_frozen_0p2_seed42  model_ckpts/b1_m3_frozen_0p2_seed42.pth \
  python train_pivot.py --model cvae-stochastic --seed 42 \
    --name b1_m3_frozen_0p2_seed42 --epochs 500 \
    --decoder-noise-init 0.2 --decoder-noise-freeze \
    --wandb-project Conditional_LV_VAE_pivot_b1

# Spectral-loss variant of the scale-conditioned VAE (model 1 family).
# weight=0.1 is the starting choice; if results are interesting, we can
# sweep weights in a follow-up.
run_b1 b1_m1_spectral_0p1_seed42  model_ckpts/b1_m1_spectral_0p1_seed42.pth \
  python train_pivot.py --model cvae --seed 42 \
    --name b1_m1_spectral_0p1_seed42 --epochs 500 \
    --use-scale-conditioning --spectral-loss-weight 0.1 \
    --wandb-project Conditional_LV_VAE_pivot_b1

log "All B1 training runs done. Running unified eval on the new checkpoints..."
ckpts=()
for v in b1_m3_frozen_0p05_seed42 b1_m3_frozen_0p1_seed42 b1_m3_frozen_0p2_seed42; do
  [[ -f "model_ckpts/${v}.pth" ]] && ckpts+=("cvae-stochastic=model_ckpts/${v}.pth")
done
[[ -f "model_ckpts/b1_m1_spectral_0p1_seed42.pth" ]] && \
  ckpts+=("cvae-scale-cond=model_ckpts/b1_m1_spectral_0p1_seed42.pth")

if (( ${#ckpts[@]} > 0 )); then
  log "Evaluating: ${ckpts[*]}"
  python analysis/evaluate_all_models.py --checkpoints "${ckpts[@]}" \
    --out RESULTS_COMPARATIVE_B1.json
  log "B1 eval done → RESULTS_COMPARATIVE_B1.json"
else
  log "No B1 checkpoints produced; skipping eval"
fi

# Step 3: hand control back to autoqueue
log "Resuming autoqueue (SIGCONT $AUTOQUEUE_PID)..."
kill -CONT "$AUTOQUEUE_PID" || log "WARNING: SIGCONT failed"
log "=== B1 watcher finished ==="
