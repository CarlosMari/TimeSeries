#!/bin/bash
# Sequential training queue. Waits for any current python training process to
# exit, then runs the next item. Designed to be left running in the background;
# logs to logs/autoqueue.log. Idempotent — skips if checkpoint exists.

cd "$(dirname "$0")/.."
source TimeSeries/bin/activate
mkdir -p logs model_ckpts

wait_for_gpu_free() {
  # Match any python process whose command line contains either trainer.
  while true; do
    local p1=$(pgrep -f "train_pivot.py" || true)
    local p2=$(pgrep -f "train_glv_regression.py" || true)
    if [[ -z "$p1" && -z "$p2" ]]; then
      return 0
    fi
    sleep 60
  done
}

run_next() {
  local label=$1; shift
  local ckpt=$1; shift
  if [[ -f "$ckpt" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $label (already $ckpt)"
    return
  fi
  wait_for_gpu_free
  echo "[$(date +%H:%M:%S)] START $label  ckpt=$ckpt  cmd=$*"
  # Run synchronously — no nohup, no `&` — so this script blocks until done.
  "$@" > "logs/${label}.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] DONE  $label  rc=$rc"
}

# -----------------------------------------------------------------------------
# Stage 1 — seed 42 for every model
# -----------------------------------------------------------------------------

run_next model_2_seed42  model_ckpts/model_2_seed42.pth \
  python train_pivot.py --model cvae --seed 42 --name model_2_seed42 --epochs 500 --no-scale-conditioning
run_next model_3_seed42  model_ckpts/model_3_seed42.pth \
  python train_pivot.py --model cvae-stochastic --seed 42 --name model_3_seed42 --epochs 500
run_next model_7_seed42  model_ckpts/model_7_seed42.pth \
  python train_glv_regression.py --seed 42 --name model_7_seed42 --epochs 200
run_next model_4_seed42  model_ckpts/model_4_seed42.pth \
  python train_pivot.py --model latent-ode --seed 42 --name model_4_seed42 --epochs 500
run_next model_5_seed42  model_ckpts/model_5_seed42.pth \
  python train_pivot.py --model transformer-vae --seed 42 --name model_5_seed42 --epochs 500
run_next model_6_seed42  model_ckpts/model_6_seed42.pth \
  python train_pivot.py --model kan-vae --seed 42 --name model_6_seed42 --epochs 500

echo "[$(date +%H:%M:%S)] === Stage-1 (seed-42) training complete ==="

# -----------------------------------------------------------------------------
# Intermediate evaluation + figure regen
# -----------------------------------------------------------------------------

wait_for_gpu_free
ckpts=()
declare -A type_for_m
type_for_m[1]=cvae-scale-cond
type_for_m[2]=cvae-no-scale-cond
type_for_m[3]=cvae-stochastic
type_for_m[4]=latent-ode
type_for_m[5]=transformer-vae
type_for_m[6]=kan-vae
for m in 1 2 3 4 5 6; do
  ckpt=model_ckpts/model_${m}_seed42.pth
  [[ -f "$ckpt" ]] && ckpts+=("${type_for_m[$m]}=$ckpt")
done
ckpt=model_ckpts/model_7_seed42.pth
[[ -f "$ckpt" ]] && ckpts+=("glv-regression=$ckpt")
echo "[$(date +%H:%M:%S)] Evaluating: ${ckpts[*]}"
python analysis/evaluate_all_models.py --checkpoints "${ckpts[@]}" --out RESULTS_COMPARATIVE.json
python analysis/make_comparative_figures.py --results RESULTS_COMPARATIVE.json

# -----------------------------------------------------------------------------
# Stage 2 — seeds 123 + 2026
# -----------------------------------------------------------------------------

for seed in 123 2026; do
  for m in 1 2 3 4 5 6; do
    case $m in
      1) model_arg=cvae;            extra="--use-scale-conditioning" ;;
      2) model_arg=cvae;            extra="--no-scale-conditioning" ;;
      3) model_arg=cvae-stochastic; extra="--use-scale-conditioning" ;;
      4) model_arg=latent-ode;      extra="--use-scale-conditioning" ;;
      5) model_arg=transformer-vae; extra="--use-scale-conditioning" ;;
      6) model_arg=kan-vae;         extra="--use-scale-conditioning" ;;
    esac
    name=model_${m}_seed${seed}
    run_next $name  model_ckpts/${name}.pth \
      python train_pivot.py --model $model_arg --seed $seed --name $name --epochs 500 $extra
  done
  run_next model_7_seed${seed}  model_ckpts/model_7_seed${seed}.pth \
    python train_glv_regression.py --seed $seed --name model_7_seed${seed} --epochs 200
done

# Final eval
wait_for_gpu_free
ckpts=()
for seed in 42 123 2026; do
  for m in 1 2 3 4 5 6; do
    ckpt=model_ckpts/model_${m}_seed${seed}.pth
    [[ -f "$ckpt" ]] && ckpts+=("${type_for_m[$m]}=$ckpt")
  done
  ckpt=model_ckpts/model_7_seed${seed}.pth
  [[ -f "$ckpt" ]] && ckpts+=("glv-regression=$ckpt")
done
echo "[$(date +%H:%M:%S)] FINAL eval: ${ckpts[*]}"
python analysis/evaluate_all_models.py --checkpoints "${ckpts[@]}" --out RESULTS_COMPARATIVE.json --force
python analysis/make_comparative_figures.py --results RESULTS_COMPARATIVE.json

echo "[$(date +%H:%M:%S)] === ALL DONE ==="
