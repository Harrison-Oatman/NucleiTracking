#!/bin/bash
#SBATCH --job-name=unwrapper          # Job name
#SBATCH --output=output_%j.out        # Standard output and error log
#SBATCH --ntasks=1                    # Number of tasks (cores)
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --mem=25G                     # Memory limit (total)
#SBATCH --partition=gen               # Partition name
#SBATCH --cpus-per-task=1             # Number of CPU cores per task

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

echo "Running unwrap script"

python NucleiTracking/batched/unwrap.py -i "/mnt/home/hoatman/ceph/ilastik_postprocessed" -o "/mnt/home/hoatman/ceph" --level INFO
