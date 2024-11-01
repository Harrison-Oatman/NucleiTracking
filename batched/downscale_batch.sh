#!/bin/bash
#SBATCH --job-name=processlightsheet         # Job name
#SBATCH --output=output_%j.out       # Standard output and error log
#SBATCH --ntasks=1                   # Number of tasks (cores)
#SBATCH --time=08:00:00               # Time limit hrs:min:sec
#SBATCH --mem=250G                     # Memory limit (total)
#SBATCH --partition=gen               # Partition name
#SBATCH --cpus-per-task=32              # Number of CPU cores per task

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/process_data.py -i "/mnt/ceph/users/hoatman/lightsheet_20241030/Raw Images" -o "/mnt/ceph/users/hoatman/lightsheet_20241030/Raw Images" --nprocs 32 --level INFO
