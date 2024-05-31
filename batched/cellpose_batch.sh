#!/bin/bash
#SBATCH --job-name=cellposebatch         # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=01:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=40G                            # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=1                             # Number of GPUs
#SBATCH --cpus-per-task=2                   # Number of CPU cores per task


# Load your Python environment if needed
module load cuda/11.8.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/multiprocess_cellpose.py -i "/mnt/home/hoatman/ceph/lightsheet2/Recon_fused_tp_104_ch_0.tif" --model 3d08 --diam 17 --stitch_threshold 0.6 --use_gpu --axes zyx --batch_size 64 --nprocs 1 --level WARN --channels 0 0