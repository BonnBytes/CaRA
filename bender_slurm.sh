#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=DIT_train
#SBATCH --output=./out/train_net-%j.out
#SBATCH --error=./out/train_net-%j.err
#SBATCH --partition=A40medium
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=23:59:59

source /home/s33zganj/.bashrc
module load Python
module load CUDA

srun torchrun --standalone --nproc_per_node 1 /home/s33zganj/CoDA/scripts/train_cifar10.py 
