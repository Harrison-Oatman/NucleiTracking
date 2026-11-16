#!/bin/bash
#SBATCH --job-name=spot_pseudolabels
#SBATCH --output=output_%j.out
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# Pseudo-masks from spots + pretrained Cellpose-SAM 3D baseline scored against the spots.
# One embryo, a handful of frames spread over NC10-NC14. Edit the block below.

set -eo pipefail   # stop at the first failing step

module load modules/2.2-20230808
module load cuda/12.1.1
source ~/miniforge3/etc/profile.d/conda.sh
export PATH="$HOME/bin:$PATH"
# Steps 1 and 3 need pandas + pytables (once: conda activate cellpose_sam; pip install tables).
# Step 2 needs cellpose >= 4 (cpsam), as in cellpose_sam_cli.sh.
PREP_ENV=cellpose_sam
CPSAM_ENV=cellpose_sam
conda activate "$PREP_ENV"

EMBRYO=20250131
BASE_DIR="/mnt/home/hoatman/ceph/lightsheet_${EMBRYO}/raw_image/downscaled"
RECON="${BASE_DIR}/recon"
SPOTS="/mnt/home/hoatman/ceph/spots/${EMBRYO}_spots.h5"     # Globus target for data/spots/*.h5
OUT="${BASE_DIR}/seg3d/pseudolabels"
SCRIPTS="$HOME/NucleiTracking/batched"   # works from any submit directory

# h5 frames (recon file = frame + 1). 20250131 cycles by frame: NC10 26-43, NC11 44-86,
# NC12 87-133, NC13 134-204, NC14 205-246. 85 is mid NC11->12 division (mitosis check).
FRAMES="30 60 85 110 170 225 245"

# 1. pseudo-masks (+ 6 planes per axis per frame for Cellpose training)
python "$SCRIPTS"/spot_pseudolabels.py \
    --recon "$RECON" --spots "$SPOTS" --out "$OUT" \
    --frames $FRAMES --slices 6

# 2. pretrained Cellpose-SAM in 3D on the same cropped volumes
conda activate "$CPSAM_ENV"
python -m cellpose \
    --dir "${OUT}/volumes" \
    --pretrained_model cpsam \
    --use_gpu --save_tif --no_npy --verbose \
    --do_3D --z_axis 0 \
    --savedir "${OUT}/cpsam_3d"
# nuclei are ~9-14 px across here; if recall is low, try adding --diameter 12

conda activate "$PREP_ENV"

# 3. score both against the spots (pseudo-masks score 1.0 by construction; it is a format check)
python "$SCRIPTS"/score_masks_vs_spots.py \
    --masks "${OUT}/cpsam_3d" --spots "${OUT}/labels" --out "${OUT}/scores_cpsam_3d.csv"

# 4. (later) fine-tune Cellpose-SAM on the 2D slices; check flags with `python -m cellpose --help`
# python -m cellpose --train --dir "${OUT}/cellpose_slices" --mask_filter _masks \
#     --pretrained_model cpsam --n_epochs 100 --learning_rate 0.00001 --weight_decay 0.1 \
#     --train_batch_size 1 --use_gpu --model_name_out cpsam_spots_${EMBRYO}
