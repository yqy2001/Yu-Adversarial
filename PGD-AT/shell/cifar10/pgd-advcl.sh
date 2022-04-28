#!/bin/bash
#SBATCH --job-name pgd-advcl
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd-advcl.out

python pgd-advcl.py --epsilon 0.031 --net 'resnet18' --advcl --advcl_weight 0.2 --out-dir ./results/$1 \
                --batch_size 512 #--lr-schedule cosine --lr-max 0.1

CUDA_VISIBLE_DEVICES='0,1' python3 autoattack_eval.py \
    --model /home/JJ_Group/yuqy/projects/SupAdv/PGD-AT/results/$1/bestpoint.pth.tar \
    --log_path ./results/aa/pgd-advcl/$1.txt \
    --save_dir ./results/aa/pgd-advcl/$1_AA_results \

# sbatch shell/cifar10/pgd-advcl.sh pgd-advcl
# sh shell/cifar10/pgd-advcl.sh pgd-advcl 0 1 | tee logs/pgd-advcl.out