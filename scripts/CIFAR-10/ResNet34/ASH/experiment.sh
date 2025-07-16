python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --ood_eval_type standard \
    --ash_p 85 \
    --ood_eval_method ASH 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --scale_threshold 1.0 \
    --ood_eval_type adaptive \
    --ash_p 75 \
    --ood_eval_method ASH  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_scale_type std \
    --scale_threshold 1.0 \
    --ood_eval_type adaptive \
    --ash_p 75 \
    --ood_eval_method ASH 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_scale_type max \
    --scale_threshold 4.0 \
    --ood_eval_type adaptive \
    --ash_p 75 \
    --ood_eval_method ASH 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_scale_type median \
    --scale_threshold 1.0 \
    --ood_eval_type adaptive \
    --ash_p 75 \
    --ood_eval_method ASH 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_scale_type entropy \
    --scale_threshold 3.0 \
    --ood_eval_type adaptive \
    --ash_p 85 \
    --ood_eval_method ASH 