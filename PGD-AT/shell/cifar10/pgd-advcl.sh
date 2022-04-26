#!/bin/bash
#SBATCH --job-name pgd-advcl
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd-advcl.out

python GAIRAT.py --epsilon 0.031 --net 'resnet18' --advcl --advcl_weight 1.0

# sbatch shell/cifar10/pgd-advcl.sh pgd-advcl
# sh shell/cifar10/pgd-advcl.sh pgd-advcl 0 1 | tee logs/pgd-advcl.out