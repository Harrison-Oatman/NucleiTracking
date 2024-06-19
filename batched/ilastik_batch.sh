#!/bin/bash
#SBATCH --job-name=processlightsheet         # Job name
#SBATCH --output=output_%j.out       # Standard output and error log
#SBATCH --ntasks=1                   # Number of tasks (cores)
#SBATCH --time=08:00:00               # Time limit hrs:min:sec
#SBATCH --mem=150G                     # Memory limit (total)
#SBATCH --partition=gen               # Partition name
#SBATCH --cpus-per-task=32              # Number of CPU cores per task

ilastik-1.4.0.post1-Linux/run_ilastik.sh --headless \
                   --project=./ceph/process_bgs.ilp \
                   --stack_along="t" \
                   --output_filename_format=./ceph/ilastik_out/{nickname}_probabilities.h5 \
                   "./ceph/lightsheet2/Recon_fused_tp_100_ch_0.tif"