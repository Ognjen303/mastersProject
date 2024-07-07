#!/bin/bash
#SBATCH --time=00:15:00                     # Max job execution time
#SBATCH --array=1-302
#SBATCH -J ALL45M                   # Job name
#SBATCH --output=./training-data/tsp5-6x6grid/all-45M/batch_%a/out_%a.out  # Redirect stdout to a file with array index
#SBATCH --error=./training-data/tsp5-6x6grid/all-45M/batch_%a/err_%a.err   # Redirect stderr to a file with array index
#SBATCH -A KRUEGER-SL2-CPU                  # Account, change to 'KRUEGER-SL3-GPU' for GPU
#SBATCH --cpus-per-task=1                   # Specify the number of CPU cores per task
#SBATCH -p icelake                          # Cluster for CPU nodes, change to 'ampere' for GPU

source ~/rds/hpc-work/my-pytorch-env/bin/activate

# The input and output filenames depend on the array index
input_file=batch_$SLURM_ARRAY_TASK_ID.txt
output_file=anitclk_solutions_$SLURM_ARRAY_TASK_ID.txt
input_folder=/home/os415/rds/hpc-work/concorde/TSP/training-data/tsp5-6x6grid/all-45M/batch_$SLURM_ARRAY_TASK_ID
# output_log_path=/home/os415/rds/hpc-work/concorde/TSP/training-data/tsp5-6x6grid/all-45M/batch_$SLURM_ARRAY_TASK_ID/log_flush.txt

# Print the current working directory
echo "Current Working Directory: $(pwd)"

echo "input_folder: $input_folder"
echo "input_file: $input_file"
echo "output_file: $output_file"

python optimal.py -i $input_file -o $output_file -ifol $input_folder
