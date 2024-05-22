#!/bin/bash
#SBATCH --job-name=3d_cellpose          # Job name
#SBATCH --output=output_%A_%a.out       # Standard output and error log
#SBATCH --array=0-3                     # Array range (one job per GPU)
#SBATCH --ntasks=4                      # Number of tasks (one per GPU)
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --mem=16G                       # Memory limit per task
#SBATCH --gres=gpu:4                    # Request 4 GPUs
#SBATCH --partition=gpu                 # Partition name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --cpus-per-task=1               # Number of CPU cores per task


# Load your Python environment if needed
module load cuda/11.8.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/multiprocess_cellpose.py -i "ceph/3d_microscopy/Embryo027-cleaned.tif" --model 3d08 --diam 10 --do_3d --use_gpu --axes tzyx --batch_size 64 --rank ${SLURM_ARRAY_TASK_ID} --nprocs ${TOTAL_JOBS}