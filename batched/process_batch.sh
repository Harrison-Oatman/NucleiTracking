#!/bin/bash
#SBATCH --job-name=processlightsheet         # Job name
#SBATCH --output=output_%A_%a.out       # Standard output and error log
#SBATCH --array=0-4                   # Array range
#SBATCH --ntasks=4                    # Number of tasks
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --mem=16G                       # Memory limit per task
#SBATCH --partition=genx                # Partition name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --cpus-per-task=1               # Number of CPU cores per task


source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/process_data.py -i "/mnt/home/dalber/ceph/Light_Sheet_Data/240131_hisRFP_10s_2/2024-01-31_142621/Processed/" -o "/mnt/home/hoatman/ceph/lightsheet2" --rank ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_NTASKS} --level INFO