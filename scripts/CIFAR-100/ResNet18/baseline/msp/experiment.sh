python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --ood_eval_type standard \
    --ood_eval_method baseline/msp

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --scale_threshold 0.3 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/msp  

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type std \
    --scale_threshold 1.3 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/msp 

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type max \
    --scale_threshold 10.0 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/msp 

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type median \
    --scale_threshold 0.1 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/msp 

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type entropy \
    --scale_threshold 3.0 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/msp