# python3 plots.py \
#     --model densenet101 \
#     --in-dataset CIFAR-10 \
#     --out-dataset iSUN

# python3 plots.py \
#     --model densenet101 \
#     --in-dataset CIFAR-100 \
#     --out-dataset iSUN

# python3 plots.py \
#     --model resnet34 \
#     --in-dataset CIFAR-10 \
#     --out-dataset iSUN

# python3 plots.py \
#     --model resnet34 \
#     --in-dataset CIFAR-100 \
#     --out-dataset iSUN

# python3 plots.py \
#     --model resnet18 \
#     --in-dataset CIFAR-10 \
#     --out-dataset iSUN

# python3 plots.py \
#     --model resnet18 \
#     --in-dataset CIFAR-100 \
#     --out-dataset iSUN


python3 plots.py \
    --model mobilenetv2_imagenet \
    --in-dataset 'ImageNet-1K' \
    --out-dataset SUN

python3 plots.py \
    --model resnet_imagenet34 \
    --in-dataset 'ImageNet-1K' \
    --out-dataset SUN

python3 plots.py \
    --model resnet_imagenet50 \
    --in-dataset 'ImageNet-1K' \
    --out-dataset SUN


