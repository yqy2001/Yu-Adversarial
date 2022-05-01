#!/bin/bash
#SBATCH --job-name pgd-advcl-only_x_ce-acc_aware-stpg
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd-advcl-only_x_ce-acc_aware-stpg.out

python pgd-advcl.py --epsilon 0.031 --net 'resnet18' --advcl --out-dir ./results/$1 \
                --batch_size 512 --only_x_ce --acc_aware --beta_cl 2. --stpg

python3 autoattack_eval.py \
    --model /home/JJ_Group/yuqy/projects/SupAdv/PGD-AT/results/$1/bestpoint.pth.tar \
    --log_path ./results/aa/pgd-advcl-only_x_ce-acc_aware-stpg/$1.txt \
    --save_dir ./results/aa/pgd-advcl-only_x_ce-acc_aware-stpg/$1_AA_results \

# sbatch shell/cifar10/pgd-advcl-only_x_ce-acc_aware-stpg.sh pgd-advcl-only_x_ce-acc_aware-stpg
# sh shell/cifar10/pgd-advcl-only_x_ce-acc_aware-stpg.sh pgd-advcl-only_x_ce-acc_aware-stpg 0 1 | tee logs/pgd-advcl-only_x_ce-acc_aware-stpg.out