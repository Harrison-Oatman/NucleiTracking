#!/bin/bash
#SBATCH --job-name=ilastikpostprocess         # Job name
#SBATCH --output=output_%j.out       # Standard output and error log
#SBATCH --ntasks=1                   # Number of tasks (cores)
#SBATCH --time=08:00:00               # Time limit hrs:min:sec
#SBATCH --mem=250G                     # Memory limit (total)
#SBATCH --partition=genx               # Partition name
#SBATCH --cpus-per-task=32              # Number of CPU cores per task

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

echo "Running ilastik postprocessing script"

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/ilastik_postprocess.py -i "/mnt/home/hoatman/ceph/ilastik_out" -o "/mnt/home/hoatman/ceph/ilastik_postprocessed" --nprocs 32 --level INFO
