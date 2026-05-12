#!/bin/bash
#SBATCH --job-name=bbox_segment              # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=08:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=250G                           # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=1                             # Number of GPUs
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cellpose

# ── Configuration ────────────────────────────────────────────────────────────

NAME="pole_cells"

BASE_DIR = "/mnt/home/hoatman/ceph/lightsheet_20250705/raw_image/downscaled"
RECONDIR= "${BASE_DIR}/recon"  # Directory containing recon files (e.g. .tif stacks)
OUTDIR="${BASE_DIR}/bbox/${NAME}"

# Bounding box in voxel coordinates of the recon volume (inclusive low, exclusive high)
Z_LO=40
Z_HI=250
Y_LO=800
Y_HI=900
X_LO=120
X_HI=320

# Cellpose options
MODEL="cpsam"           # pretrained model name or path to custom model
GPU=0

# ── Step 1: Crop ─────────────────────────────────────────────────────────────

echo "Step 1: cropping recon files to bounding box"

python NucleiTracking/batched/bbox_crop.py \
    -i "$RECONDIR" \
    -o "$OUTDIR" \
    --name "$NAME" \
    --z_lo $Z_LO --z_hi $Z_HI \
    --y_lo $Y_LO --y_hi $Y_HI \
    --x_lo $X_LO --x_hi $X_HI \
    --nprocs $SLURM_CPUS_PER_TASK

# ── Step 2: Cellpose 3D segmentation ─────────────────────────────────────────

echo "Step 2: running cellpose 3D segmentation"

mkdir -p "${OUTDIR}/cellpose"

python -m cellpose \
    --dir "${OUTDIR}/crops" \
    --pretrained_model "$MODEL" \
    --use_gpu \
    --gpu $GPU \
    --save_tif \
    --no_npy \
    --verbose \
    --do_3d \
    --z_axis 0 \
    --savedir "${OUTDIR}/cellpose"

# Alternative: use stitch_threshold instead of --do_3d for z-by-z stitching:
# --stitch_threshold 0.25  (remove --do_3d and --z_axis if using this)

# ── Step 3: Regionprops → centroids CSV ──────────────────────────────────────

echo "Step 3: running regionprops"

python NucleiTracking/batched/bbox_regionprops.py \
    -o "$OUTDIR"

echo "Done. Output in ${OUTDIR}/centroids_3d.csv"
