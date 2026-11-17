#!/bin/bash
#SBATCH --job-name=segemb
#SBATCH --output=segemb_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --array=0-247%4

# 3D segmentation of every recon frame of one embryo with the fine-tuned Cellpose model,
# one frame per array task (task id = frame = index in the naturally sorted recon dir).
# Resources: 1 H100 + 4 CPUs per task, %4 -> at most 4 GPUs / 16 CPUs at once. No --exclusive.
# H100 nodes: feature `h100` covers SXM5 (workergpu201-224) and PCIe (workergpu151-169); keep
# --gpus=1 untyped because the GRES type differs (h100 vs h100_pcie). To also allow
# workergpu170 (H100 PCIe without the h100 tag): --constraint="h100|sapphirerapids".
# SXM only: --constraint=h100-sxm5.
# --partition=gpu,gpuxl lets SLURM use either H100 pool (gpuxl = H100-94GB genoa) if you have
# gpuxl access. RTX6000 Pro Blackwell (-C rtxblackwell) needs a cu128 torch build (cu126 lacks
# sm_120): check python -c "import torch; print(torch.cuda.get_arch_list())" in cpbench first.
#
# Array size must match the number of recon frames (20250131: 248 -> 0-247):
#   N=$(ls /mnt/home/hoatman/ceph/lightsheet_20250625/raw_image/downscaled/recon/recon_fused_tp_*_ch_0.tif | wc -l)
#   jid=$(sbatch --parsable --array=0-$((N-1))%4 --export=ALL,EMBRYO=20250625 ~/NucleiTracking/batched/segment_embryo.sh)
# Merge into one table once all tasks end:
#   sbatch --dependency=afterany:$jid --partition=genx --time=00:30:00 --mem=32G --cpus-per-task=1 \
#       --job-name=segemb_merge --output=segemb_merge_%j.out \
#       --wrap "source ~/miniforge3/etc/profile.d/conda.sh; conda activate cpbench; \
#               python ~/NucleiTracking/batched/segment_embryo.py --merge --embryo 20250625 \
#               --recon /mnt/home/hoatman/ceph/lightsheet_20250625/raw_image/downscaled/recon \
#               --out /mnt/home/hoatman/ceph/lightsheet_20250625/raw_image/downscaled/seg3d/cpsam_v2_spots_v1"
# Finished frames are skipped, so resubmitting is safe.

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cpbench
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

EMBRYO=${EMBRYO:-20250131}
MODEL=${MODEL:-/mnt/home/hoatman/ceph/models/cpsam_v2_spots_v1}
RECON="/mnt/home/hoatman/ceph/lightsheet_${EMBRYO}/raw_image/downscaled/recon"
OUT="/mnt/home/hoatman/ceph/lightsheet_${EMBRYO}/raw_image/downscaled/seg3d/$(basename "$MODEL")"
SCRIPTS="$HOME/NucleiTracking/batched"

I=${SLURM_ARRAY_TASK_ID:?run as an array job}
echo "embryo $EMBRYO frame $I model $MODEL on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "$SCRIPTS"/segment_embryo.py --recon "$RECON" --out "$OUT" --model "$MODEL" --index "$I"
