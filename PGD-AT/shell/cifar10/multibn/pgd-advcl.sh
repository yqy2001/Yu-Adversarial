#!/bin/bash
#SBATCH --job-name pgd-advcl
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd-advcl.out

python GAIRAT.py --epsilon 0.031 --net 'resnet18' --advcl --advcl_weight 0.2 --out-dir ./results/$1 \
                --batch_size 512 #--lr-schedule cosine --lr-max 0.1

# sbatch shell/multibn/pgd-advcl.sh pgd-advcl
# sh shell/multibn/pgd-advcl.sh pgd-advcl 0 1 | tee logs/pgd-advcl.out