#!/bin/bash
# M8 heteroscedastic-decoder VAE: train 3 seeds then eval 3-lens × 3 distributions.
#
# Architectural test of the smoothness-attractor framing (PROJECT.md §1.3.0.2):
# does training under per-(t,c) Gaussian NLL produce trajectories that match
# real observation noise *without* post-hoc σ injection?

set -u
cd "$(dirname "$0")/.."
source TimeSeries/bin/activate
mkdir -p logs results/legacy

LOG=logs/m8_train_and_eval.log
exec >> "$LOG" 2>&1

date_stamp() { date -u +'[%Y-%m-%d %H:%M:%S UTC]'; }
log() { echo "$(date_stamp) $*"; }

log "=== M8 train + eval pipeline ==="

# ---- Train 3 seeds ----
for seed in 42 123 2026; do
  name="m8_seed${seed}"
  ckpt="model_ckpts/${name}.pth"
  if [[ -f "$ckpt" ]]; then
    log "SKIP $name (ckpt exists)"
    continue
  fi
  log "TRAIN $name (500 epochs, scale-cond, NLL loss)"
  python train_pivot.py --model cvae-heteroscedastic \
    --seed $seed --name $name --epochs 500 \
    --wandb-project Conditional_LV_VAE_pivot_m8 \
    > "logs/${name}.log" 2>&1
  rc=$?
  log "DONE $name rc=$rc"
done

# ---- Build ckpt list for eval ----
build_ckpts() {
  local ckpts=()
  for seed in 42 123 2026; do
    local p=model_ckpts/m8_seed${seed}.pth
    if [[ -f "$p" ]]; then
      ckpts+=("cvae-heteroscedastic=$p")
    fi
  done
  echo "${ckpts[@]}"
}

# ---- Eval on all 3 distributions at N_CHAOS=1000 (paper headline N) ----
for tset in TEST_FINAL_NOSORT TEST_OOD_Exp1 TEST_OOD_Exp5; do
  out_suffix="${tset#TEST_}"
  out="results/legacy/RESULTS_M8_${out_suffix}.json"
  log "Eval on $tset → $out"
  python analysis/evaluate_all_models.py \
    --checkpoints $(build_ckpts) \
    --test-path "data/${tset}.pkl" \
    --out "$out" --force \
    --n-chaos 1000 2>&1
done

log "=== M8 pipeline FINISHED ==="
