#!/bin/bash
#SBATCH --job-name=segemb
#SBATCH --output=segemb_%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4

# 3D segmentation of all recon frames of ONE embryo in ONE job (1 GPU, 4 CPUs): the model is
# loaded once and frames run serially (~2.2 min/frame on an H100 -> ~9-10 h for ~250 frames),
# then the per-embryo table is merged at the end. Finished frames are skipped, so a job that
# hits the time limit can simply be resubmitted.
#
# All four WT embryos, one job each (4 jobs = 4 GPUs, 16 CPUs):
#   for E in 20250131 20250625 20250702 20250704; do
#     sbatch --job-name=seg_$E --export=ALL,EMBRYO=$E ~/NucleiTracking/batched/segment_embryo.sh
#   done
# Faster: split an embryo over N jobs with SHARD=i and NSHARDS=N (each does every N-th frame;
# the job that finishes last writes the complete merged table, earlier ones a partial one):
#   for i in 0 1; do sbatch --job-name=seg_${E}_$i --export=ALL,EMBRYO=$E,SHARD=$i,NSHARDS=2 ...; done
# Other model: --export=ALL,EMBRYO=$E,MODEL=/path/to/model
#
# GPU choice: feature `h100` = SXM5 (workergpu201-224) + PCIe (workergpu151-169), all in the gpu
# partition; --gpus=1 stays untyped (GRES type differs: h100 vs h100_pcie). Add workergpu170:
# --constraint="h100|sapphirerapids"; SXM only: h100-sxm5. Blackwell: -C rtxblackwell with
# --mem-per-cpu instead of --mem, and a cu128 torch (check torch.cuda.get_arch_list() for sm_120).

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cpbench
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

EMBRYO=${EMBRYO:-20250131}
MODEL=${MODEL:-/mnt/home/hoatman/ceph/models/cpsam_v2_spots_v3}
RECON="/mnt/home/hoatman/ceph/lightsheet_${EMBRYO}/raw_image/downscaled/recon"
OUT="/mnt/home/hoatman/ceph/lightsheet_${EMBRYO}/raw_image/downscaled/seg3d/$(basename "$MODEL")"
SCRIPTS="$HOME/NucleiTracking/batched"
TILE_NORM=${TILE_NORM:-128}   # local normalization block size (0 = whole volume)
SHARD_ARGS=()
if [[ -n "$NSHARDS" ]]; then SHARD_ARGS=(--shard "${SHARD:-0}" "$NSHARDS"); fi

N=$(ls "$RECON"/recon_fused_tp_*_ch_0.tif | wc -l)
echo "embryo $EMBRYO: $N recon frames, model $MODEL, shard ${SHARD:-all}/${NSHARDS:-1} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "$SCRIPTS"/segment_embryo.py --recon "$RECON" --out "$OUT" --model "$MODEL" --tile-norm "$TILE_NORM" "${SHARD_ARGS[@]}"
python "$SCRIPTS"/segment_embryo.py --recon "$RECON" --out "$OUT" --merge --embryo "$EMBRYO"
