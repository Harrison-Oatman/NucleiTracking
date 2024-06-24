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
    local file=$1  # $1 is the first argument passed to the function, which is the file path
    local output_file="${OUTPUT_DIR}/$(basename ${file%.*})_probabilities.h5"  # Construct the output file path
    echo "Processing file: $file"  # Print the file being processed for debugging
    # Run ilastik on the file
    # Construct the ilastik command
    local ilastik_command="LAZYFLOW_TOTAL_RAM_MB=25000 ilastik-1.4.0.post1-Linux/run_ilastik.sh --headless \
                            --project=$PROJECT_FILE \
                            --stack_along=\"t\" \
                            --output_filename_format=$output_file \
                            $file"

    # Print the command being executed for debugging purposes
    echo "Executing: $ilastik_command"

    # Run the ilastik command
    eval $ilastik_command
}

export -f process_file  # Export the function so it can be used by GNU Parallel

# Path to the locally downloaded GNU Parallel script
PARALLEL=~/parallel

# Run up to 10 ilastik instances simultaneously
echo "$FILES" | tr ' ' '\n' | $PARALLEL -j 10 process_file
