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
# python train_val.py
# python fact_cp.py

python fact_cp.py --lr=1e-3 --s=1 --dim=2
python fact_cp.py --lr=1e-3 --s=10 --dim=2
python fact_cp.py --lr=1e-3 --s=0.1 --dim=2

python fact_cp.py --lr=1e-3 --s=1 --dim=4
python fact_cp.py --lr=1e-3 --s=10 --dim=4
python fact_cp.py --lr=1e-3 --s=0.1 --dim=4

python fact_cp.py --lr=1e-3 --s=1 --dim=8
python fact_cp.py --lr=1e-3 --s=10 --dim=8
python fact_cp.py --lr=1e-3 --s=0.1 --dim=8

python fact_cp.py --lr=1e-3 --s=1 --dim=16
python fact_cp.py --lr=1e-3 --s=10 --dim=16
python fact_cp.py --lr=1e-3 --s=0.1 --dim=16

python fact_cp.py --lr=1e-3 --s=1 --dim=32
python fact_cp.py --lr=1e-3 --s=10 --dim=32
python fact_cp.py --lr=1e-3 --s=0.1 --dim=32

python fact_cp.py --lr=1e-3 --s=1 --dim=64
python fact_cp.py --lr=1e-3 --s=10 --dim=64
python fact_cp.py --lr=1e-3 --s=0.1 --dim=64
