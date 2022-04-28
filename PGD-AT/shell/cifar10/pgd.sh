#!/bin/bash
#SBATCH --job-name pgd
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd.out

python GAIRAT.py --epsilon 0.031 --net 'resnet18' --out-dir ./results/$1 \
                --batch_size 512 #--lr-schedule cosine --lr-max 0.1

# sbatch shell/cifar10/pgd.sh pgd
# sh shell/cifar10/pgd.sh pgd 0 1 | tee logs/pgd.out