# python3 ReAct.py \
#     --epoch 0 \
#     --pool avg \
#     --checkpoint model \
#     --embedding_dim 2048 \
#     --model resnet_imagenet50 \
#     --in-dataset ImageNet-1K \
#     --id_loc datasets/in-imagenet/val \

# python3 ReAct.py \
#     --epoch 0 \
#     --pool avg+std \
#     --checkpoint model \
#     --embedding_dim 2048 \
#     --model resnet_imagenet50 \
#     --in-dataset ImageNet-1K \
#     --id_loc datasets/in-imagenet/val \

# python3 ReAct.py \
#     --epoch 0 \
#     --pool max \
#     --checkpoint model \
#     --embedding_dim 2048 \
#     --model resnet_imagenet50 \
#     --in-dataset ImageNet-1K \
#     --id_loc datasets/in-imagenet/val \

# python3 ReAct.py \
#     --epoch 0 \
#     --pool avg \
#     --checkpoint model \
#     --embedding_dim 2048 \
#     --model resnet_imagenet34 \
#     --in-dataset ImageNet-1K \
#     --id_loc datasets/in-imagenet/val \

# python3 ReAct.py \
#     --epoch 0 \
#     --pool avg+std \
#     --checkpoint model \
#     --embedding_dim 2048 \
#     --model resnet_imagenet34 \
#     --in-dataset ImageNet-1K \
#     --id_loc datasets/in-imagenet/val \

# python3 ReAct.py \
#     --epoch 0 \
#     --pool max \
#     --checkpoint model \
#     --embedding_dim 2048 \
#     --model resnet_imagenet34 \
#     --in-dataset ImageNet-1K \
#     --id_loc datasets/in-imagenet/val \

python3 ReAct.py \
    --epoch 0 \
    --pool avg \
    --checkpoint model \
    --embedding_dim 1280 \
    --model mobilenetv2_imagenet \
    --in-dataset ImageNet-1K \
    --id_loc datasets/in-imagenet/val \

python3 ReAct.py \
    --epoch 0 \
    --pool avg+std \
    --checkpoint model \
    --embedding_dim 1280 \
    --model mobilenetv2_imagenet \
    --in-dataset ImageNet-1K \
    --id_loc datasets/in-imagenet/val \

python3 ReAct.py \
    --epoch 0 \
    --pool max \
    --checkpoint model \
    --embedding_dim 1280 \
    --model mobilenetv2_imagenet \
    --in-dataset ImageNet-1K \
    --id_loc datasets/in-imagenet/val \