#!/bin/bash
#SBATCH --job-name=segmulti
#SBATCH --output=segmulti_%j.out
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=14:00:00

# ONE submission, 4 GPU workers that may land on DIFFERENT H100 nodes (1 GPU + 4 CPUs + 48 GB
# each), so it does not wait for a node with 4 free GPUs. Worker i (= SLURM_PROCID) does frames
# i, i+4, i+8, ... of every embryo; then the batch script merges each embryo's table.
# Finished frames are skipped: if the time limit hits, resubmit the same command to resume.
# Time: 1219 frames / 4 workers ~ 305 frames x ~2.2 min ~ 11-12 h -> 14 h limit.
# All workers start together once 4 single GPUs are free somewhere (no backfill per worker);
# for the most scheduling freedom use segment_embryo.sh, one 1-GPU job per embryo.
#
# Submit:   sbatch ~/NucleiTracking/batched/segment_embryos_multi.sh
# Subset:   sbatch --export=ALL,EMBRYOS="20250625 20250704" ~/NucleiTracking/batched/segment_embryos_multi.sh
# Monitor:  tail -f segmulti_<jobid>_w*.log ; squeue -j <jobid> -o "%.10i %.8T %.25R %.20S"

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cpbench

export EMBRYOS=${EMBRYOS:-"20250131 20250625 20250702 20250704"}
export MODEL=${MODEL:-/mnt/home/hoatman/ceph/models/cpsam_v2_spots_v3}
export SCRIPTS="$HOME/NucleiTracking/batched"
export TILE_NORM=${TILE_NORM:-128}   # local normalization block size (0 = whole volume)
export NW=${SLURM_NTASKS:-4}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
recon() { echo "/mnt/home/hoatman/ceph/lightsheet_$1/raw_image/downscaled/recon"; }
out() { echo "/mnt/home/hoatman/ceph/lightsheet_$1/raw_image/downscaled/seg3d/$(basename "$MODEL")"; }

echo "embryos: $EMBRYOS | model: $MODEL | workers: $NW | nodes: $SLURM_JOB_NODELIST"
for E in $EMBRYOS; do
    n=$(ls "$(recon "$E")"/recon_fused_tp_*_ch_0.tif | wc -l)
    echo "$E: $n frames"
    if (( n == 0 )); then echo "no recon frames for $E"; exit 1; fi
done

# each srun task gets its own GPU; it loops over embryos with shard = its rank
srun --ntasks="$NW" --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --kill-on-bad-exit=0 \
    bash -c '
    i=$SLURM_PROCID
    exec > "segmulti_${SLURM_JOB_ID}_w${i}.log" 2>&1
    echo "worker $i on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    for E in $EMBRYOS; do
        python "$SCRIPTS"/segment_embryo.py \
            --recon "/mnt/home/hoatman/ceph/lightsheet_$E/raw_image/downscaled/recon" \
            --out "/mnt/home/hoatman/ceph/lightsheet_$E/raw_image/downscaled/seg3d/$(basename "$MODEL")" \
            --model "$MODEL" --shard "$i" "$NW" --tile-norm "$TILE_NORM"
    done' || echo "a worker failed; see segmulti_${SLURM_JOB_ID}_w*.log (merging what is done)"

for E in $EMBRYOS; do
    python "$SCRIPTS"/segment_embryo.py --recon "$(recon "$E")" --out "$(out "$E")" --merge --embryo "$E"
done
