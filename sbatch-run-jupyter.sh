#!/bin/bash

#SBATCH --job-name jupyter
#SBATCH --output log-jupyter.txt
#SBATCH --time 0-12

#SBATCH -p gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 3

source .env/bin/activate
python -m jupyter notebook --ip=*

