#!/bin/bash
#SBATCH --job-name pgd
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd.out

python pgd-advcl.py --epsilon 0.031 --net 'resnet18' --out-dir ./results/$1 \
                --batch_size 512 #--lr-schedule cosine --lr-max 0.1

CUDA_VISIBLE_DEVICES='0,1' python3 autoattack_eval.py \
    --model /home/JJ_Group/yuqy/projects/SupAdv/PGD-AT/results/$1/bestpoint.pth.tar \
    --log_path ./results/aa/pgd-advcl/$1.txt \
    --save_dir ./results/aa/pgd-advcl/$1_AA_results \

# sbatch shell/cifar10/pgd.sh pgd
# sh shell/cifar10/pgd.sh pgd 0 1 | tee logs/pgd.out