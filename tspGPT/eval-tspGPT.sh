#!/bin/bash
#SBATCH --time=24:00:00                 # Max job execution time
#SBATCH -J BC-30%                      # Job name
#SBATCH --out=./outputs-final-countdown/final-anticlk/microGPT/final-report/BC-30%-cont-2.out    # redirect stdout to a file
#SBATCH --err=.outputs-final-countdown/final-anticlk/microGPT/final-report/BC-30%-cont-2.err    # redirect stderr to a file
#SBATCH -A KRUEGER-SL2-GPU              # Account, change to 'KRUEGER-SL3-CPU' for CPU
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -p ampere                       # cluster for GPU nodes, change to 'icelake' for CPU


source ~/rds/hpc-work/my-pytorch-env/bin/activate

python main-diff-eval.py -lr 1e-3 -wd 1e-4 -i BC-swaps/train-swapBC-30%.txt -e BC-swaps/eval.txt -ebc BC-swaps/eval-swapped-BC.txt -eran BC-swaps/eval-random.txt -o report-final-microGPT-BC-swap-30% -ch 9500 -epochs 20 -trglen 4 -msize 7