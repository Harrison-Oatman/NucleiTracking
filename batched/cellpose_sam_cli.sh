#!/bin/bash
#SBATCH --job-name=cellpose                  # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=08:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=750G                            # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=3                             # Number of GPUs
#SBATCH --cpus-per-task=6                    # Number of CPU cores per task

echo "starting task"

# Load Python environment
module load modules/2.2-20230808
module load cuda/12.1.1
source ~/miniforge3/etc/profile.d/conda.sh
export PATH="$HOME/bin:$PATH"
conda activate cellpose_sam

TOPDIR="/mnt/home/hoatman/ceph/lightsheet_20250414/raw_image/downscaled/uv_unwrap/"
SAVEDIR="${TOPDIR}/cellpose_output"

export SAVEDIR

mkdir -p "$SAVEDIR"

# GPU list (0,1,2)
GPUS=(0 1 2)

run_cellpose_dir_2d() {
    dir=$1
    jobnum=$2
    gpu=$(( (jobnum - 1) % 3 ))
    python -m cellpose --dir "$dir" --pretrained_model uv_sam_001 --use_gpu --save_tif --verbose --norm_percentile 0 100 --no_npy --savedir "${dir}/cellpose" --gpu $gpu --channel_axis 2
}
export -f run_cellpose_dir_2d

# Run on *loc.tif in top-level directory
echo "Running cellpose.py on cellpose_stack folders"
find "$TOPDIR" -mindepth 2 -maxdepth 2 -name 'cellpose_stack' -type d | \
    parallel --jobs 3 --ungroup --env run_cellpose_dir_2d --env SAVEDIR \
    'run_cellpose_dir_2d {} {#}'


#run_cellpose_dir() {
#    dir=$1
#    jobnum=$2
#    gpu=$(( (jobnum - 1) % 3 ))
#    python -m cellpose --dir "$dir" --pretrained_model uv_sam_001 --use_gpu --save_tif --verbose --norm_percentile 0 100 --no_npy --savedir "$SAVEDIR" --gpu $gpu --z_axis 0 --channel_axis 3 --stitch_threshold 0.25
#}
#export -f run_cellpose_dir
#
## Run in each subdirectory of TOPDIR
#echo "Running cellpose.py in subdirectories..."
#find "$TOPDIR" -mindepth 2 -maxdepth 2 -name 'vals' -type d | \
#    parallel --jobs 3 --ungroup --env run_cellpose_dir --env SAVEDIR \
#    'run_cellpose_dir {} {#}'


