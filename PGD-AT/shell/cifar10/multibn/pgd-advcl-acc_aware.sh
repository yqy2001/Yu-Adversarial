#!/bin/bash
#SBATCH --job-name pgd-advcl-acc_aware
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd-advcl-acc_aware.out

python GAIRAT.py --epsilon 0.031 --net 'resnet18' --advcl --advcl_weight 1.0 \
                --acc_aware --beta_cl 2. --out-dir ./results/$1 \
                --batch_size 512 #--lr-schedule cosine --lr-max 0.1

# sbatch shell/cifar10/multibn/pgd-advcl-acc_aware.sh pgd-advcl-acc_aware
# sh shell/cifar10/multibn/pgd-advcl-acc_aware.sh pgd-advcl-acc_aware 0 1 | tee logs/pgd-advcl-acc_aware.out