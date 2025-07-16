python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --ood_eval_type standard \
    --threshold 1.2 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type avg \
    --scale_threshold 0.1 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type std \
    --scale_threshold 0.5 \
    --ood_eval_type adaptive \
    --threshold 1.1 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type max \
    --scale_threshold 10.0 \
    --ood_eval_type adaptive \
    --threshold 1.1 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_scale_type median \
    --scale_threshold 2.0 \
    --ood_eval_type adaptive \
    --threshold 1.1 \
    --ood_eval_method ReAct 

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
    --threshold 1.1 \
    --ood_eval_method ReAct