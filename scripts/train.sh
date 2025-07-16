python3 train.py \
 --epochs 100 \
 --batch-size 64 \
 --learning_rate 0.1 \
 --embedding_dim 512 \
 --optimizer_type sgd \
 --id_loc datasets/in/ \
 --in-dataset CIFAR-10  \
 --model resnet18 \

python3 train.py \
 --epochs 100 \
 --batch-size 64 \
 --learning_rate 0.1 \
 --embedding_dim 512 \
 --optimizer_type sgd \
 --id_loc datasets/in/ \
 --in-dataset CIFAR-100  \
 --model resnet18 \

python3 train.py \
 --epochs 100 \
 --batch-size 64 \
 --learning_rate 0.1 \
 --embedding_dim 512 \
 --optimizer_type sgd \
 --id_loc datasets/in/ \
 --in-dataset CIFAR-10  \
 --model resnet34 \

python3 train.py \
 --epochs 100 \
 --batch-size 64 \
 --learning_rate 0.1 \
 --embedding_dim 512 \
 --optimizer_type sgd \
 --id_loc datasets/in/ \
 --in-dataset CIFAR-100  \
 --model resnet34 \

python3 train.py \
 --epochs 100 \
 --batch-size 64 \
 --learning_rate 0.1 \
 --embedding_dim 342 \
 --optimizer_type sgd \
 --id_loc datasets/in/ \
 --in-dataset CIFAR-10  \
 --model densenet101 \

python3 train.py \
 --epochs 100 \
 --batch-size 64 \
 --learning_rate 0.1 \
 --embedding_dim 342 \
 --optimizer_type sgd \
 --id_loc datasets/in/ \
 --in-dataset CIFAR-100  \
 --model densenet101 \

