#!/bin/bash
#SBATCH --job-name=uv_unwrap         # Job name
#SBATCH --output=output_%j.out          # Standard output and error log
#SBATCH --ntasks=1                      # Number of tasks (cores)
#SBATCH --time=08:00:00                 # Time limit hrs:min:sec
#SBATCH --mem=250G                      # Memory limit (total)
#SBATCH --partition=genx                # Partition name
#SBATCH --cpus-per-task=128              # Number of CPU cores per task

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/uv_unwrap.py -i "/mnt/home/hoatman/ceph/lightsheet_20250625/raw_image/downscaled/recon" --obj "/mnt/home/hoatman/ceph/lightsheet_20250625/raw_image/downscaled/uv_maps/" --nprocs 128 --range "-12" 8 21
