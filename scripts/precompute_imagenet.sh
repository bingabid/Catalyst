
# python3 precompute.py \
#  --pool avg \
#  --embedding_dim 2048 \
#  --in-dataset ImageNet-1K \
#  --model resnet_imagenet50 \
#  --id_loc datasets/in-imagenet/val \

# python3 precompute.py \
#  --pool avg \
#  --embedding_dim 2048 \
#  --in-dataset ImageNet-1K \
#  --model resnet_imagenet34 \
#  --id_loc datasets/in-imagenet/val \

# python3 precompute.py \
#  --pool avg \
#  --embedding_dim 2048 \
#  --in-dataset ImageNet-1K \
#  --model resnet_imagenet18 \
#  --id_loc datasets/in-imagenet/val \

python3 precompute.py \
 --pool avg \
 --embedding_dim 1280 \
 --in-dataset ImageNet-1K \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
