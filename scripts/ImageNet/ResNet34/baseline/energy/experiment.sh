python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet34 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type avg \
    --ood_eval_type standard \
    --ood_eval_method baseline/energy 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet34 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type avg \
    --scale_threshold 1.0 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/energy

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet34 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type std \
    --scale_threshold 1.0 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/energy 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet34 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type max \
    --scale_threshold 4.0 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/energy 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet34 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type median \
    --scale_threshold 0.1 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/energy 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet34 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type entropy \
    --scale_threshold 5.0 \
    --ood_eval_type adaptive \
    --ood_eval_method baseline/energy 