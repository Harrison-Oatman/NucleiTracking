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
python NucleiTracking/batched/uv_unwrap.py -i "/mnt/home/hoatman/ceph/lightsheet_trk_20250318/raw_image/downscaled/recon" --obj "/mnt/home/hoatman/ceph/lightsheet_trk_20250318/raw_image/downscaled/recon/mesh_with_UV.obj" --nprocs 16 --range "-2" 2 5
