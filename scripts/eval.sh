python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --ood_eval_type standard \
    --ood_eval_method baseline/msp