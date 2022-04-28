
CUDA_VISIBLE_DEVICES='0,1' python3 autoattack_eval.py \
    --model='/home/JJ_Group/yuqy/projects/SupAdv/PGD-AT/results/multibn/pgd-advcl/bestpoint.pth.tar' \
    --log_path='./results/aa/pgd-advcl/pgd-advcl.txt' \
    --save_dir='./results/aa/pgd-advcl/pgd_advcl_AA_results' \
    --model_type 'resnet18_multibn'

CUDA_VISIBLE_DEVICES='0,1' python3 autoattack_eval.py \
    --model='/home/JJ_Group/yuqy/projects/SupAdv/PGD-AT/results/multibn/pgd/bestpoint.pth.tar' \
    --log_path='./results/aa/pgd/pgd.txt' \
    --save_dir='./results/aa/pgd/pgd_AA_results'