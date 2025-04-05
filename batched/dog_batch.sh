#!/bin/bash
#SBATCH --job-name=dog         # Job name
#SBATCH --output=output_%j.out       # Standard output and error log
#SBATCH --ntasks=1                   # Number of tasks (cores)
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --mem=250G                     # Memory limit (total)
#SBATCH --partition=genx               # Partition name
#SBATCH --cpus-per-task=32              # Number of CPU cores per task

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/dog.py -i "/mnt/home/hoatman/ceph/lightsheet_trk_20250325/raw_image/downscaled/recon/" --nprocs 32 --min_distance 3 --threshold_abs 25 --sigma_low 2 --sigma_high 7 --level INFO
