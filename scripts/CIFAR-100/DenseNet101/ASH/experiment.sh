python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --ood_eval_type standard \
    --ash_p 90 \
    --ood_eval_method ASH 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --scale_threshold 0.01 \
    --ood_eval_type adaptive \
    --ash_p 90 \
    --ood_eval_method ASH  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type std \
    --scale_threshold 0.01 \
    --ood_eval_type adaptive \
    --ash_p 90 \
    --ood_eval_method ASH  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type max \
    --scale_threshold 0.1 \
    --ood_eval_type adaptive \
    --ash_p 90 \
    --ood_eval_method ASH    

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type median \
    --scale_threshold 0.1 \
    --ood_eval_type adaptive \
    --ash_p 90 \
    --ood_eval_method ASH    

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type entropy \
    --scale_threshold 5.0 \
    --ood_eval_type adaptive \
    --ash_p 90 \
    --ood_eval_method ASH  