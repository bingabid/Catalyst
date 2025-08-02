python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2_imagenet \
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
    --model mobilenetv2_imagenet \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type avg \
    --scale_threshold 0.2 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2_imagenet \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type avg \
    --scale_threshold 0.3 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2_imagenet \
    --id_loc datasets/in-imagenet/val \
    --in-dataset ImageNet-1K \
    --ood_loc datasets/ood-imagenet/ \
    --ood_scale_type avg \
    --scale_threshold 0.4 \
    --ood_eval_type adaptive \
    --threshold 1.0 \
    --ood_eval_method ReAct



# python3 eval_ood.py \
#     --score msp \
#     --batch-size 64 \
#     --model densenet_imagenet121 \
#     --id_loc datasets/in-imagenet/val \
#     --in-dataset ImageNet-1K \
#     --ood_loc datasets/ood-imagenet/ \
#     --ood_scale_type avg \
#     --scale_threshold 1.5 \
#     --ood_eval_type adaptive \
#     --ood_eval_method baseline/msp 
