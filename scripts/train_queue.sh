#!/bin/bash
# Sequential training queue for the 7-model pivot.
#
# Usage:  ./scripts/train_queue.sh [stage]
# Stages:
#   1     models 1, 2, 3, 7 (seed 42 first; LSTM-VAE family + GLV regression)
#   2     model 4 (Latent-ODE), model 5 (Transformer-VAE), model 6 (KAN-VAE) — seed 42
#   3     remaining seeds (123, 2026) across all 7 models
#   all   run all stages
#
# Each step writes its log to logs/<name>.log. Skips if a checkpoint already
# exists (idempotent). Designed to be safe to re-run if interrupted.
#
# 2026-05-15 — start with stage 1, evaluate, then expand if numbers look good.

set -e
cd "$(dirname "$0")/.."
source TimeSeries/bin/activate
mkdir -p logs model_ckpts

run() {
  local name=$1
  local cmd=$2
  if [[ -f "model_ckpts/${name}.pth" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $name (checkpoint exists)"
    return
  fi
  echo "[$(date +%H:%M:%S)] START $name"
  eval "$cmd" 2>&1 | tee "logs/${name}.log"
  echo "[$(date +%H:%M:%S)] DONE $name"
}

# Run a no-cond / cond variant of cvae
cvae_run() {
  local model=$1 seed=$2 epochs=$3 cond=$4 name=$5 extra=$6
  local flag=$([[ "$cond" == "yes" ]] && echo "--use-scale-conditioning" || echo "--no-scale-conditioning")
  run "$name" "python train_pivot.py --model $model --seed $seed --name $name --epochs $epochs $flag $extra"
}

stage_1() {
  cvae_run cvae             42 500 yes   model_1_seed42         # model 1 (already running externally)
  cvae_run cvae             42 500 no    model_2_seed42         # model 2
  cvae_run cvae-stochastic  42 500 yes   model_3_seed42         # model 3
  run "model_7_seed42" "python train_glv_regression.py --seed 42 --name model_7_seed42 --epochs 200"
}

stage_2() {
  cvae_run latent-ode       42 500 yes   model_4_seed42         # model 4
  cvae_run transformer-vae  42 500 yes   model_5_seed42         # model 5
  cvae_run kan-vae          42 500 yes   model_6_seed42         # model 6
}

stage_3() {
  for seed in 123 2026; do
    cvae_run cvae             $seed 500 yes   model_1_seed${seed}
    cvae_run cvae             $seed 500 no    model_2_seed${seed}
    cvae_run cvae-stochastic  $seed 500 yes   model_3_seed${seed}
    cvae_run latent-ode       $seed 500 yes   model_4_seed${seed}
    cvae_run transformer-vae  $seed 500 yes   model_5_seed${seed}
    cvae_run kan-vae          $seed 500 yes   model_6_seed${seed}
    run "model_7_seed${seed}" "python train_glv_regression.py --seed $seed --name model_7_seed${seed} --epochs 200"
  done
}

case "${1:-all}" in
  1) stage_1 ;;
  2) stage_2 ;;
  3) stage_3 ;;
  all) stage_1 ; stage_2 ; stage_3 ;;
  *) echo "unknown stage $1" ; exit 1 ;;
esac
