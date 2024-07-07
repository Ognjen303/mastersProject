#!/bin/bash
#SBATCH --time=35:30:00                 # Max job execution time
#SBATCH -J three-GPT2                      # Job name
#SBATCH --out=./outputs-final-countdown/final-anticlk/GPT2/report-final-GPT2-run3-2x2-three.out    # redirect stdout to a file
#SBATCH --err=.outputs-final-countdown/final-anticlk/GPT2/report-final-GPT2-run3-2x2-three.err    # redirect stderr to a file
#SBATCH -A KRUEGER-SL2-GPU              # Account, change to 'KRUEGER-SL3-CPU' for CPU
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -p ampere                       # cluster for GPU nodes, change to 'icelake' for CPU


source ~/rds/hpc-work/my-pytorch-env/bin/activate

python main-diff-eval.py -lr 1e-4 -wd 1e-5 -i 2x2-centre/train-no-centre.txt -e 2x2-centre/eval-three-city.txt -o report-final-GPT2-run3-2x2-three -epochs 10 -trglen 4 -msize 117