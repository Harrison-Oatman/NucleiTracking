#!/bin/bash
#SBATCH --job-name=ls_batch                  # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=01:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=64G                            # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=4                             # Number of GPUs
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task

echo "starting task"

# Load your Python environment if needed
module load cuda/11.8.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/multiprocess_cellpose_dir.py -i "ceph/lightsheet2" --model 3d08 --diam 10 --do_3d --use_gpu --axes zyx --batch_size 64 --nprocs 4 --level WARN