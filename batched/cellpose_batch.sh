#!/bin/bash
#SBATCH --job-name=cellposebatch         # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=01:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=40G                            # Memory limit
#SBATCH --partition=gpu                      # Partition name
#SBATCH --gpus=1                             # Number of GPUs
#SBATCH --cpus-per-task=9                  # Number of CPU cores per task


# Load your Python environment if needed
module load modules/2.2-20230808
module load cuda/11.8.0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cellpose

# Run your Python script with job_number and total_jobs
python NucleiTracking/batched/multiprocess_cellpose.py -i "/mnt/home/hoatman/ceph/lightsheet_trk_20250319a/raw_image/downscaled/uv_unwrap/large_all_vals.tif" --model uv_005 --diam 12 --use_gpu --axes tyx --batch_size 64 --nprocs 8 --level WARN --channels 0 0