#!/bin/bash
#SBATCH --job-name=ls_batch                  # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=08:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=750G                            # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=4                             # Number of GPUs
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task

echo "starting task"

# Load Python environment
module load modules/2.2-20230808
module load cuda/11.8.0
source ~/miniforge3/etc/profile.d/conda.sh
export PATH="$HOME/bin:$PATH"
conda activate cellpose

TOPDIR="/mnt/home/hoatman/ceph/lightsheet_20250206/raw_image/downscaled/uv_unwrap/"
SAVEDIR="${TOPDIR}/cellpose_output"

export SAVEDIR

mkdir -p "$SAVEDIR"

# GPU list (0,1,2)
GPUS=(0 1 2)

run_cellpose_file() {
    file=$1
    jobnum=$2
    gpu=$(( (jobnum - 1) % 3 ))
    echo "Processing $file on GPU $gpu"
    echo "$SAVEDIR"
    python -m cellpose --image_path "$file" --pretrained_model uv_007 --diameter 11.54 --use_gpu --save_tif --verbose --norm_percentile 0 100 --no_npy --savedir "$SAVEDIR" --gpu $gpu
}
export -f run_cellpose_file

# Run on *loc.tif in top-level directory
echo "Running cellpose.py on top-level *vals.tif files..."
find "$TOPDIR" -maxdepth 1 -type f -name '*vals.tif' | \
    parallel --jobs 3 --ungroup --env run_cellpose_file --env SAVEDIR \
    'run_cellpose_file {} {#}'