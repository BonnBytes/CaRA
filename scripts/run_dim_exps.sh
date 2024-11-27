#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=mnist
#SBATCH --output=./dims-%j.out
#SBATCH --error=./dims-%j.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00
#SBATCH --mem=4gb
echo "job started."
source ~/.bashrc
conda activate codatorch
export TQDM_MININTERVAL=5

python dim_experiment.py --dataset smallnorb_azi --dim 5 --rank 1
python dim_experiment.py --dataset smallnorb_azi --dim 4 --rank 1
python dim_experiment.py --dataset smallnorb_azi --dim 3 --rank 1
python dim_experiment.py --dataset smallnorb_azi --dim 2 --rank 1
python dim_experiment.py --dataset smallnorb_azi --dim 1 --rank 1

python dim_experiment.py --dataset smallnorb_azi --dim 5 --rank 2
python dim_experiment.py --dataset smallnorb_azi --dim 4 --rank 2
python dim_experiment.py --dataset smallnorb_azi --dim 3 --rank 2
python dim_experiment.py --dataset smallnorb_azi --dim 2 --rank 2
python dim_experiment.py --dataset smallnorb_azi --dim 1 --rank 2


python dim_experiment.py --dataset smallnorb_azi --dim 5 --rank 4
python dim_experiment.py --dataset smallnorb_azi --dim 4 --rank 4
python dim_experiment.py --dataset smallnorb_azi --dim 3 --rank 4
python dim_experiment.py --dataset smallnorb_azi --dim 2 --rank 4
python dim_experiment.py --dataset smallnorb_azi --dim 1 --rank 4


python dim_experiment.py --dataset smallnorb_azi --dim 5 --rank 8
python dim_experiment.py --dataset smallnorb_azi --dim 4 --rank 8
python dim_experiment.py --dataset smallnorb_azi --dim 3 --rank 8
python dim_experiment.py --dataset smallnorb_azi --dim 2 --rank 8
python dim_experiment.py --dataset smallnorb_azi --dim 1 --rank 8

python dim_experiment.py --dataset smallnorb_azi --dim 5 --rank 16
python dim_experiment.py --dataset smallnorb_azi --dim 4 --rank 16
python dim_experiment.py --dataset smallnorb_azi --dim 3 --rank 16
python dim_experiment.py --dataset smallnorb_azi --dim 2 --rank 16
python dim_experiment.py --dataset smallnorb_azi --dim 1 --rank 16

python dim_experiment.py --dataset smallnorb_azi --dim 5 --rank 24
python dim_experiment.py --dataset smallnorb_azi --dim 4 --rank 24
python dim_experiment.py --dataset smallnorb_azi --dim 3 --rank 24
python dim_experiment.py --dataset smallnorb_azi --dim 2 --rank 24
python dim_experiment.py --dataset smallnorb_azi --dim 1 --rank 24

python dim_experiment.py --dataset smallnorb_azi --dim 5 --rank 32
python dim_experiment.py --dataset smallnorb_azi --dim 4 --rank 32
python dim_experiment.py --dataset smallnorb_azi --dim 3 --rank 32
python dim_experiment.py --dataset smallnorb_azi --dim 2 --rank 32
python dim_experiment.py --dataset smallnorb_azi --dim 1 --rank 32


echo "done."