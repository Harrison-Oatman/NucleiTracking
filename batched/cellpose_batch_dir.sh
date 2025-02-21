#!/bin/bash
#SBATCH --job-name=ls_batch                  # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=08:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=250G                            # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=4                             # Number of GPUs
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task

echo "starting task"

# Load your Python environment if needed
module load modules/2.2-20230808
module load cuda/11.8.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/multiprocess_cellpose_dir.py -i "ceph/lightsheet_20250206/raw_image/downscaled/recon" --model lightsheet002 --diam 11 -c "-1.5" -t 99.5 --use_gpu -f 1.5 --flow3D_smooth 1.0 --do_3d --axes zyx --batch_size 512 --nprocs 4 --level INFO