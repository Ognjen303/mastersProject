#!/bin/bash
#SBATCH --time=35:30:00                 # Max job execution time
#SBATCH -J base-9                      # Job name
#SBATCH --out=./outputs-final-countdown/final-anticlk/GPT2/report-final-BASELINE-GPT2-9.out    # redirect stdout to a file
#SBATCH --err=.outputs-final-countdown/final-anticlk/GPT2/report-final-BASELINE-GPT2-9.err    # redirect stderr to a file
#SBATCH -A KRUEGER-SL2-GPU              # Account, change to 'KRUEGER-SL3-CPU' for CPU
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -p ampere                       # cluster for GPU nodes, change to 'icelake' for CPU


source ~/rds/hpc-work/my-pytorch-env/bin/activate

python main-diff-eval.py -lr 1e-4 -wd 1e-5 -i train.txt -e eval.txt -o report-final-BASELINE-GPT2-9 -epochs 10 -trglen 4 -msize 117