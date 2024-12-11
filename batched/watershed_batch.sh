#!/bin/bash
#SBATCH --job-name=watershed         # Job name
#SBATCH --output=output_%j.out       # Standard output and error log
#SBATCH --ntasks=1                   # Number of tasks (cores)
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --mem=250G                     # Memory limit (total)
#SBATCH --partition=genx               # Partition name
#SBATCH --cpus-per-task=32              # Number of CPU cores per task

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/watershed.py -i "/mnt/home/hoatman/ceph/lightsheet_20241104/raw_image/downscaled/recon2/" --nprocs 32 --min_distance 2 --seed_threshold 20 --sigma_low 2 --sigma_high 6 --level INFO
