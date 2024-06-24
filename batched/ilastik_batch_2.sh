#!/bin/bash
#SBATCH --job-name=ilastik                   # Job name
#SBATCH --output=output_%j.out               # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks (cores)
#SBATCH --time=08:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=250G                           # Memory limit (total)
#SBATCH --partition=gen                      # Partition name
#SBATCH --cpus-per-task=16                    # Number of CPU cores per task

# Directory containing the files
FILE_DIR="./ceph/lightsheet2"
OUTPUT_DIR="./ceph/ilastik_out"
PROJECT_FILE="./ceph/process_bgs.ilp"

# Find all files matching the pattern and store them in an array
FILES=$(find $FILE_DIR -name 'Recon_fused_tp_*_ch_0.tif')

# Function to process each file
process_file() {
    local file=$1
    local output_file="${OUTPUT_DIR}/$(basename ${file%.*})_probabilities.h5"
    LAZYFLOW_TOTAL_RAM_MB=25000 ilastik-1.4.0.post1-Linux/run_ilastik.sh --headless \
                   --project=$PROJECT_FILE \
                   --stack_along="t" \
                   --output_filename_format=$output_file \
                   $file
}

export -f process_file

# Run up to 10 ilastik instances simultaneously
echo "$FILES" | tr ' ' '\n' | parallel -j 10 process_file
