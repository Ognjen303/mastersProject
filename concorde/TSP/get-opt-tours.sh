#!/bin/bash
#SBATCH --time=01:50:00                     # Max job execution time
#SBATCH -J 30M                   # Job name
#SBATCH --array=1-300                        # Number of tasks in the array
#SBATCH --output=./training-data/tsp5-6x6grid/batch_%a/batch_%a.out  # Redirect stdout to a file with array index
#SBATCH --error=./training-data/tsp5-6x6grid/batch_%a/batch_%a.err   # Redirect stderr to a file with array index
#SBATCH -A KRUEGER-SL3-CPU                  # Account, change to 'KRUEGER-SL3-GPU' for GPU
#SBATCH --cpus-per-task=1                   # Specify the number of CPU cores per task
#SBATCH --nodes=1                           # Request 1 node
#SBATCH -p icelake                          # Cluster for CPU nodes, change to 'ampere' for GPU

source ~/rds/hpc-work/my-pytorch-env/bin/activate

# The input and output filenames depend on the array index
input_file="batch_${SLURM_ARRAY_TASK_ID}.txt"
output_file="solutions_${SLURM_ARRAY_TASK_ID}.txt"
input_folder="./training-data/tsp5-6x6grid/batch_${SLURM_ARRAY_TASK_ID}"

python get-opt-tours.py -i $input_file -o $output_file -ifol $input_folder
