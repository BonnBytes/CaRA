#!/bin/bash
#
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --time=71:59:59
#SBATCH --mem=515000

ml CUDA
source /home/lveerama/.bashrc
conda activate pytorch


# python train_val.py --trainable-ranks=2
# python train_val.py --trainable-ranks=5
python train_val.py --trainable-ranks=9
