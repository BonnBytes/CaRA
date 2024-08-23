#!/bin/bash
#
#SBATCH -A hpca
#SBATCH --nodes=1
#SBATCH --job-name=diff_train
#SBATCH --output=./out/diff_train-%j.out
#SBATCH --error=./out/diff_train-%j.err
#SBATCH --time=23:59:59
#SBATCH --gres gpu:1
#SBATCH --partition sgpu_medium


module load CUDA
source ~/.bashrc
source activate CPDeApprox

export PYTHONPATH=.

python scripts/train_cifar10.py --epochs=20 --batch-size 512 