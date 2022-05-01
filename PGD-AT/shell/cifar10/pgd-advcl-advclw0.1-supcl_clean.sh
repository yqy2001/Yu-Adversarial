#!/bin/bash
#SBATCH --job-name pgd-advcl-advclw0.1-supcl_clean
#SBATCH --gres=gpu:a100:2
#SBATCH --time 1-10:00:00
#SBATCH --output pgd-advcl-advclw0.1-supcl_clean.out

python pgd-advcl.py --epsilon 0.031 --net 'resnet18' --advcl --advcl_weight 0.1 --out-dir ./results/$1 \
                --batch_size 512 --supcl_clean

python3 autoattack_eval.py \
    --model /home/JJ_Group/yuqy/projects/SupAdv/PGD-AT/results/$1/bestpoint.pth.tar \
    --log_path ./results/aa/pgd-advcl-advclw0.1-supcl_clean/$1.txt \
    --save_dir ./results/aa/pgd-advcl-advclw0.1-supcl_clean/$1_AA_results \

# sbatch shell/cifar10/pgd-advcl-advclw0.1-supcl_clean.sh pgd-advcl-advclw0.1-supcl_clean
# sh shell/cifar10/pgd-advcl-advclw0.1-supcl_clean.sh pgd-advcl-advclw0.1-supcl_clean 0 1 | tee logs/pgd-advcl-advclw0.1-supcl_clean.out