python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet50 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type avg \
    --ood_eval_type standard \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet50 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type avg \
    --scale_threshold 0.1 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet50 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type std \
    --scale_threshold 0.15 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet50 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type max \
    --scale_threshold 1.0 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet50 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type median \
    --scale_threshold 0.1 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet_imagenet50 \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type entropy \
    --scale_threshold 4.0 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct

# python3 eval_ood.py \
#     --score energy \
#     --batch-size 64 \
#     --model resnet_imagenet50 \
#     --id_loc datasets/in-imagenet/val \
#     --in-dataset ImageNet-1K \
#     --ood_loc datasets/ood-imagenet/ \
#     --ood_scale_type feature_entropy \
#     --ood_eval_type adaptive \
#     --ood_eval_method ReAct 