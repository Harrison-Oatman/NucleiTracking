#!/bin/bash
#SBATCH --job-name=convert_hdf5         # Job name
#SBATCH --output=output_%j.out          # Standard output and error log
#SBATCH --ntasks=1                      # Number of tasks (cores)
#SBATCH --time=08:00:00                 # Time limit hrs:min:sec
#SBATCH --mem=250G                      # Memory limit (total)
#SBATCH --partition=genx                # Partition name
#SBATCH --cpus-per-task=16              # Number of CPU cores per task

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/hdf5_extraction.py -i "/mnt/ceph/users/lyang/For_Harrison/2025-07-05_155602/raw/" -o "/mnt/home/hoatman/ceph/lightsheet_20250705/raw_image" --nprocs 16 --level INFO

2025-07-04_112724
2025-07-05_155602