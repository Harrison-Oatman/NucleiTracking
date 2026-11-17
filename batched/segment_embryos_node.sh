#!/bin/bash
#SBATCH --job-name=segnode
#SBATCH --output=segnode_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=h100
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=16:00:00

# ONE job, 4 GPUs on one H100 node: 3D segmentation of every recon frame of several embryos.
# Worker i (GPU i, 4 CPUs) loads the model once and does frames i, i+4, i+8, ... of each embryo
# in turn; when all workers finish, each embryo's table is merged. Finished frames are skipped,
# so resubmitting the same command resumes where it stopped.
# Time: 248 + 333 + 312 + 326 = 1219 frames x ~2.2 min / 4 GPUs ~ 11 h (+ model loading);
# 16 h leaves margin. Check the partition limit with: sinfo -p gpu -o %l
# If the job hits the limit, resubmit the same command: done frames are skipped.
#
# Submit:   sbatch ~/NucleiTracking/batched/segment_embryos_node.sh
# Other set: sbatch --export=ALL,EMBRYOS="20250131 20250625" ~/NucleiTracking/batched/segment_embryos_node.sh
# Monitor:  tail -f segnode_<jobid>_w*.log     (one log per worker, one line per frame)
#           ls /mnt/home/hoatman/ceph/lightsheet_<E>/raw_image/downscaled/seg3d/cpsam_v2_spots_v1/centroids | wc -l
# Faster (more GPUs): one job per embryo with segment_embryo.sh (1 GPU each), or submit this
# script once per embryo with EMBRYOS=<one embryo> (4 GPUs each).
#
# H100 nodes: `h100` = SXM5 (workergpu201-224, 4 GPUs) + PCIe (workergpu151-169, 8 GPUs).

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cpbench
export PYTHONUNBUFFERED=1

EMBRYOS=${EMBRYOS:-"20250131 20250625 20250702 20250704"}
MODEL=${MODEL:-/mnt/home/hoatman/ceph/models/cpsam_v2_spots_v1}
NW=${NWORKERS:-4}
SCRIPTS="$HOME/NucleiTracking/batched"
recon() { echo "/mnt/home/hoatman/ceph/lightsheet_$1/raw_image/downscaled/recon"; }
out() { echo "/mnt/home/hoatman/ceph/lightsheet_$1/raw_image/downscaled/seg3d/$(basename "$MODEL")"; }

echo "embryos: $EMBRYOS | model: $MODEL | workers: $NW | $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
for E in $EMBRYOS; do
    n=$(ls "$(recon "$E")"/recon_fused_tp_*_ch_0.tif | wc -l)
    echo "$E: $n frames"
    if (( n == 0 )); then echo "no recon frames for $E"; exit 1; fi
done

worker() {
    local i=$1
    for E in $EMBRYOS; do
        CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
            python "$SCRIPTS"/segment_embryo.py --recon "$(recon "$E")" --out "$(out "$E")" \
            --model "$MODEL" --shard "$i" "$NW"
    done
}

pids=()
for ((i = 0; i < NW; i++)); do
    worker "$i" > "segnode_${SLURM_JOB_ID:-local}_w$i.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && echo "a worker failed; see segnode_${SLURM_JOB_ID:-local}_w*.log (merging what is done)"

for E in $EMBRYOS; do
    python "$SCRIPTS"/segment_embryo.py --recon "$(recon "$E")" --out "$(out "$E")" --merge --embryo "$E"
done
exit $fail
