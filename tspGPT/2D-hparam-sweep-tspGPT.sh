#!/bin/bash
#SBATCH --array=0-15                       # Number of tasks in the array (3 learning rates * 3 weight decays)
#SBATCH --time=04:00:00                 # Max job execution time
#SBATCH -J tsp6-nano                 # Job name
#SBATCH --out=./outputs-feb/tsp6-nano-2Dsweep-%a-%A.out    # redirect stdout to a file
#SBATCH --err=./outputs-feb/tsp6-nano-2Dsweep-%a-%A.err    # redirect stderr to a file
#SBATCH -A KRUEGER-SL3-GPU              # Account, change to 'KRUEGER-SL3-CPU' for CPU
#SBATCH --cpus-per-task=4               # Specify the number of CPU cores per task
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1                      # Request 1 task
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -p ampere                       # cluster for GPU nodes, change to 'icelake' for CPU

# Define an array of learning rates and weight decays
learning_rates=(1e-3 1e-4 1e-5 1e-6)
weight_decays=(1e-3 1e-4 1e-5 1e-6)

# Calculate the indices for learning rates and weight decays
lr_index=$((SLURM_ARRAY_TASK_ID / ${#weight_decays[@]}))
wd_index=$((SLURM_ARRAY_TASK_ID % ${#weight_decays[@]}))

# Set the learning rate and weight decay for this task
learning_rate=${learning_rates[$lr_index]}
weight_decay=${weight_decays[$wd_index]}

source ~/rds/hpc-work/my-pytorch-env/bin/activate

# Create a directory for the current combination of learning rate and weight decay
output_dir="tsp6-nano-lr-${learning_rate}-wd-${weight_decay}"
mkdir -p $output_dir

python main.py -lr $learning_rate -wd $weight_decay -i solutions-tsp6-10M.txt -o $output_dir -trglen 5 -msize 3

deactivate
