#!/bin/bash
#SBATCH --job-name=cpbench
#SBATCH --output=cpbench_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-151%4

# Resource cap: each task = 1 GPU + 2 CPUs (not a whole node: no --exclusive), and %4 lets at
# most 4 tasks run at once -> at most 4 GPUs and 8 CPUs in use. Keep %4 on any --array override.
# Cellpose model benchmark vs tracked spots, one (model, tp) per array task.
# Task i -> model = MODELS[i / N_TPS], tp = TPS[i % N_TPS]  (model-major).
# Finished tasks (tp_XXX_row.csv present) are skipped, so resubmitting is safe.
#
# Submit (from any directory) and merge when all tasks end:
#   jid=$(sbatch --parsable ~/NucleiTracking/batched/benchmark_models_vs_spots.sh)
#   sbatch --dependency=afterany:$jid --partition=genx --time=00:20:00 --mem=16G --cpus-per-task=1 \
#       --job-name=cpbench_sum --output=cpbench_sum_%j.out \
#       --wrap "source ~/miniforge3/etc/profile.d/conda.sh; conda activate cpbench; \
#               python ~/NucleiTracking/batched/benchmark_models_vs_spots.py \
#               --out /mnt/home/hoatman/ceph/lightsheet_20250131/raw_image/downscaled/seg3d/benchmark --summarize"
#
# Timepoints: default = every 6th tp from 87 to 303, plus 307 -> 38 tps x 4 models = 152 tasks
# (--array=0-151). Spots cover h5 frames 26-246 = tp 87-307 (h5 frame = tp - 61).
# All tps (221 x 4 = 884 tasks; check MaxArraySize with `scontrol show config | grep MaxArray`):
#   sbatch --array=0-883%4 --export=ALL,TPS="$(seq -s ' ' 87 307)" ~/NucleiTracking/batched/benchmark_models_vs_spots.sh
# Any list: --export=ALL,TPS="96 130 220 307" with --array=0-$((4*N-1))%4.
# Fewer models: --export=ALL,MODELS="cpsam_v2 cpdino-vitb" (and resize --array).
#
# Env (once): python 3.11; torch 2.9.1 + torchvision 0.24.1 for the node's CUDA;
#   pip install "cellpose==4.2.1.1" "git+https://github.com/facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751" tables
# Weights (once, on a login node; ~3.8 GB into ~/.cellpose/models):
#   python -c "from cellpose import models; [models.CellposeModel(gpu=False, pretrained_model=m) for m in ['cpsam','cpsam_v2','cpdino','cpdino-vitb']]"

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cpbench
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

EMBRYO=20250131
RECON="/mnt/home/hoatman/ceph/lightsheet_${EMBRYO}/raw_image/downscaled/recon"
SPOTS="/mnt/home/hoatman/ceph/spots/${EMBRYO}_spots.h5"
OUT="/mnt/home/hoatman/ceph/lightsheet_${EMBRYO}/raw_image/downscaled/seg3d/benchmark"
SCRIPTS="$HOME/NucleiTracking/batched"
OFFSET=61   # h5 frame = tp - OFFSET, for files named recon_fused_tp_<tp>_ch_0.tif

MODELS=${MODELS:-"cpsam cpsam_v2 cpdino cpdino-vitb"}
TPS=${TPS:-"$(seq -s ' ' 87 6 303) 307"}

read -ra M <<< "$MODELS"
read -ra T <<< "$TPS"
N=$(( ${#M[@]} * ${#T[@]} ))
I=${SLURM_ARRAY_TASK_ID:?run as an array job}
if (( I >= N )); then
    echo "task $I >= ${#M[@]} models x ${#T[@]} tps = $N; nothing to do"
    exit 0
fi
echo "task $I/$N: model ${M[I / ${#T[@]}]} tp ${T[I % ${#T[@]}]} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "$SCRIPTS"/benchmark_models_vs_spots.py \
    --recon-dir "$RECON" --spots "$SPOTS" --out "$OUT" --offset "$OFFSET" \
    --models $MODELS --tps $TPS --task-index "$I"
