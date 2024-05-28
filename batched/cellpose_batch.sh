#!/bin/bash
#SBATCH --job-name=cellposebatch         # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=01:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=64G                            # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=4                             # Number of GPUs
#SBATCH --cpus-per-task=4                    # Number of CPU cores per task


# Load your Python environment if needed
module load cuda/11.8.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/multiprocess_cellpose.py -i "ceph/3d_microscopy/tll_Embryo002b_bgs.tif" --model 3d08 --diam 17 --stitch_threshold 0.6 --use_gpu --axes tzcyx --batch_size 64 --nprocs 4 --level WARN --channels 2 0