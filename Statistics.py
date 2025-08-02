import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from configs import ModelConfig
from utils.utils import set_model, get_device
from experiments.model.sobel import SobelDepthwise
from utils.datasets import load_in_dataset, load_out_dataset, dataset_loader

def args_parser():
    parser = argparse.ArgumentParser(description='Analyze Distribution',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--p', default=None, type=int, help='DICE pruning level')
    parser.add_argument('--ash_p', default=None, type=int, help='ASH pruning level')
    parser.add_argument('--threshold', default=None, type=float, help='ReAct threshold')
    parser.add_argument('--epoch', default ="100", type=str, help='which epoch to test')
    parser.add_argument('--std', default=1, type=float, help='how many stndard deviation ')
    parser.add_argument('--ood_eval', action='store_true', help='false for extracting statistics')

    parser.add_argument('--embedding_dim', default = 512, type=int, help='penultimate feature dim')
    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--ood_loc', default="datasets/ood/", type=str, help='location of ood datasets')
    parser.add_argument('--out-dataset', default="iSUN-dummy", type=str, help='out-distribution dataset')
    parser.add_argument('--id_loc', default="datasets/in/", type=str, help='location of in-distribution dataset')
    
    parser.add_argument('--batch-size', default= 256, type=int, help='mini-batch size')
    parser.add_argument('--checkpoint', default = 'model', type=str, help='checkpoint name')
    parser.add_argument('--normalize', action='store_true', help='normalize feat embeddings')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in in-dataset')
    parser.add_argument('--model', default='densenet101', type=str, help='model architecture: [ resnet18, resnet34, densenet101]')

    device = get_device()
    parser.add_argument('--device', type=torch.device, default=device, help = 'device type for accelerated training')

    args = parser.parse_args()

    if args.in_dataset in ["CIFAR-10"]:
        args.num_classes = 10
    elif args.in_dataset in ["CIFAR-100"]:
        args.num_classes = 100
    elif args.in_dataset in ["ImageNet-1K"]:
        args.num_classes = 1000
    return args

def setup_directory(args):

    # base directory
    base_directory_name = f"{args.model}/{args.in_dataset}/"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)
    model_statistics_directory = os.path.join(ModelConfig.model_statistics_directory, base_directory_name)

    # load checkpoints
    model_checkpoint_name = f"{args.checkpoint}.pt"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)
    args.ckpt = os.path.join(model_checkpoint_directory, model_checkpoint_name)
    

    if not os.path.exists(model_statistics_directory):
        os.makedirs(model_statistics_directory)
    return model_statistics_directory

def obtain_in_statistics(args, model, data_set, batch_size = None):
    if batch_size is None:
        batch_size = args.batch_size
    device = args.device
    data_size = len(data_set)
    feature_dim = model.dim_in
    num_classes = args.num_classes

    avg_pool = nn.AdaptiveAvgPool2d((1,1))
    max_pool = nn.AdaptiveMaxPool2d((1,1))

    labels = np.zeros(data_size)
    avg = np.zeros((data_size, feature_dim))
    std = np.zeros((data_size, feature_dim))
    maxi = np.zeros((data_size, feature_dim))
    logit = np.zeros((data_size, num_classes))
    median = np.zeros((data_size, feature_dim))
    entropy = np.zeros((data_size, feature_dim))

    data_loader = dataset_loader(args, data_set, batch_size=batch_size)
    # model.eval() # already set to eval during setting up model

    with torch.inference_mode():
        for batch_idx, (x, y) in tqdm(enumerate(data_loader)):

            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, data_size)

            x = x.to(device)
            y = y.to(device)

            # get logits:
            batch_logit = model(x)
            logit[start_ind:end_ind, :] = batch_logit.data.cpu().numpy()

            # get statistics: avg, std, max
            activation_map = model.encoder(x)              #[batch_size, feature_dim, height, width]
            batch_avg = avg_pool(activation_map)           #[batch_size, feature_dim, 1, 1]
            batch_avg = batch_avg.view(-1, feature_dim)    #[batch_size, feature_dim]
            batch_std = activation_map.std(dim = (2, 3))   #[batch_size, feature_dim]
            batch_max = max_pool(activation_map)           #[batch_size, feature_dim, 1, 1]
            batch_max = batch_max.view(-1, feature_dim)    #[batch_size, feature_dim]
            

            labels[start_ind:end_ind]  = y.data.cpu().numpy()
            avg[start_ind:end_ind, :]  = batch_avg.data.cpu().numpy()
            std[start_ind:end_ind, :]  = batch_std.data.cpu().numpy()
            maxi[start_ind:end_ind, :] = batch_max.data.cpu().numpy()
            

            # get statistics: median, entropy, sobel
            b, d, w, h = activation_map.shape
            activation_map_flat = activation_map.view(b, d, -1) #[batch_size, feature_dim, w*h]
            batch_median = activation_map_flat.median(dim=2).values
            probs = F.softmax(activation_map_flat, dim=-1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            median[start_ind:end_ind, :] = batch_median.data.cpu().numpy()
            entropy[start_ind:end_ind, :] = batch_entropy.data.cpu().numpy()

    return labels, avg, std, maxi, median, entropy, logit

def obtain_out_statistics(args, model, data_set, batch_size = None):
    if batch_size is None:
        batch_size = args.batch_size
    device = args.device
    data_size = len(data_set)
    feature_dim = model.dim_in
    num_classes = args.num_classes

    avg_pool = nn.AdaptiveAvgPool2d((1,1))
    max_pool = nn.AdaptiveMaxPool2d((1,1))

    avg = np.zeros((data_size, feature_dim))
    std = np.zeros((data_size, feature_dim))
    maxi = np.zeros((data_size, feature_dim))
    median = np.zeros((data_size, feature_dim))
    logit = np.zeros((data_size, num_classes))
    entropy = np.zeros((data_size, feature_dim))

    data_loader = dataset_loader(args, data_set, batch_size=batch_size)
    # model.eval() # already set to eval during setting up model

    with torch.inference_mode():
        for batch_idx, (x,y) in tqdm(enumerate(data_loader)):
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, data_size)

            x = x.to(device)

            # get logits:
            batch_logit = model(x)
            logit[start_ind:end_ind, :] = batch_logit.data.cpu().numpy()

            # get statistics: avg, std, max
            activation_map = model.encoder(x)              #[batch_size, feature_dim, height, width]
            batch_avg = avg_pool(activation_map)           #[batch_size, feature_dim, 1, 1]
            batch_avg = batch_avg.view(-1, feature_dim)    #[batch_size, feature_dim]
            batch_std = activation_map.std(dim = (2, 3))   #[batch_size, feature_dim]
            batch_max = max_pool(activation_map)           #[batch_size, feature_dim, 1, 1]
            batch_max = batch_max.view(-1, feature_dim)    #[batch_size, feature_dim]

            avg[start_ind:end_ind, :]  = batch_avg.data.cpu().numpy()
            std[start_ind:end_ind, :]  = batch_std.data.cpu().numpy()
            maxi[start_ind:end_ind, :] = batch_max.data.cpu().numpy()
            

            #median and entropy
            b, d, w, h = activation_map.shape
            activation_map_flat = activation_map.view(b, d, -1)
            batch_median = activation_map_flat.median(dim=2).values
            probs = F.softmax(activation_map_flat, dim=-1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            median[start_ind:end_ind, :] = batch_median.data.cpu().numpy()
            entropy[start_ind:end_ind, :] = batch_entropy.data.cpu().numpy()
       
    return avg, std, maxi, median, entropy, logit

def save_weights(args, model, weight_directory):
    W_fname = weight_directory + 'W.npy'
    b_fname = weight_directory + 'b.npy'
    if args.model in ['densenet101', 'resnet18', 'resnet34']: 
        W = model.output_layer.weight.detach().cpu().numpy()
        b = model.output_layer.bias.detach().cpu().numpy()
    elif args.model in ['resnet_imagenet34', 'resnet_imagenet50']:
        W = model.fc.weight.detach().cpu().numpy()
        b = model.fc.bias.detach().cpu().numpy()
    elif args.model in ['mobilenetv2_imagenet']:
        W = model.classifier[1].weight.detach().cpu().numpy()
        b = model.classifier[1].bias.detach().cpu().numpy()
    elif args.model in [ 'densenet_imagenet121']:
        W = model.classifier.weight.detach().cpu().numpy()
        b = model.classifier.bias.detach().cpu().numpy()
    np.save(W_fname, W)
    np.save(b_fname, b)

def save_in_features(in_features_directory, in_scales_directory, avg, std, maxi, median, entropy, logit, labels):
    # save featuures
    std_fname = in_features_directory + 'std.npy'
    max_fname = in_features_directory + 'max.npy'
    avg_fname = in_features_directory + 'avg.npy'
    logit_fname = in_features_directory  + 'logit.npy'
    labels_fname = in_features_directory + 'labels.npy'
    median_fname = in_features_directory + 'median.npy'
    entropy_fname = in_features_directory + 'entropy.npy'

    np.save(avg_fname, avg)
    np.save(std_fname, std)
    np.save(max_fname, maxi)
    np.save(logit_fname, logit)
    np.save(labels_fname, labels)
    np.save(median_fname, median)
    np.save(entropy_fname, entropy)


    # compute and save scales
    avg_scale = avg.sum(axis = 1)
    std_scale = std.sum(axis = 1) 
    max_scale = maxi.sum(axis = 1)
    logit_scale   = logit.sum(axis = 1)
    median_scale  = median.sum(axis = 1) 
    entropy_scale = entropy.sum(axis = 1)

    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    feature_entropy_prob = softmax(avg, axis=1)
    feature_entropy_scale = -np.sum(feature_entropy_prob * np.log2(feature_entropy_prob + 1e-10), axis=1)

    scale_avg_fname = in_scales_directory + 'avg.npy'
    scale_std_fname = in_scales_directory + 'std.npy'
    scale_max_fname = in_scales_directory + 'max.npy'
    scale_logit_fname = in_scales_directory  + 'logit.npy'
    scale_median_fname = in_scales_directory + 'median.npy'
    scale_entropy_fname = in_scales_directory + 'entropy.npy'
    scale_feature_entropy_fname = in_scales_directory + 'feature_entropy.npy'

    np.save(scale_avg_fname, avg_scale)
    np.save(scale_std_fname, std_scale)
    np.save(scale_max_fname, max_scale)
    np.save(scale_logit_fname, logit_scale)
    np.save(scale_median_fname, median_scale)
    np.save(scale_entropy_fname, entropy_scale)
    np.save(scale_feature_entropy_fname, feature_entropy_scale)

def save_out_features(out_features_directory, out_scales_directory, avg, std, maxi, median, entropy, logit):
    # save featuures
    std_fname = out_features_directory + 'std.npy'
    max_fname = out_features_directory + 'max.npy'
    avg_fname = out_features_directory + 'avg.npy'
    logit_fname = out_features_directory  + 'logit.npy'
    median_fname = out_features_directory + 'median.npy'
    entropy_fname = out_features_directory + 'entropy.npy'

    np.save(avg_fname, avg)
    np.save(std_fname, std)
    np.save(max_fname, maxi)
    np.save(logit_fname, logit)
    np.save(median_fname, median)
    np.save(entropy_fname, entropy)

    # compute and save scales
    avg_scale = avg.sum(axis = 1)
    std_scale = std.sum(axis = 1) 
    max_scale = maxi.sum(axis = 1)
    logit_scale   = logit.sum(axis = 1)
    median_scale  = median.sum(axis = 1) 
    entropy_scale = entropy.sum(axis = 1)

    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    feature_entropy_prob = softmax(avg, axis=1)
    feature_entropy_scale = -np.sum(feature_entropy_prob * np.log2(feature_entropy_prob + 1e-10), axis=1)

    scale_avg_fname = out_scales_directory + 'avg.npy'
    scale_std_fname = out_scales_directory + 'std.npy'
    scale_max_fname = out_scales_directory + 'max.npy'
    scale_logit_fname = out_scales_directory  + 'logit.npy'
    scale_median_fname = out_scales_directory + 'median.npy'
    scale_entropy_fname = out_scales_directory + 'entropy.npy'
    scale_feature_entropy_fname = out_scales_directory + 'feature_entropy.npy'

    np.save(scale_avg_fname, avg_scale)
    np.save(scale_std_fname, std_scale)
    np.save(scale_max_fname, max_scale)
    np.save(scale_logit_fname, logit_scale)
    np.save(scale_median_fname, median_scale)
    np.save(scale_entropy_fname, entropy_scale)
    np.save(scale_feature_entropy_fname, feature_entropy_scale)

     
def main(args):

    # setting up statistics directory
    model_statistics_directory = setup_directory(args)
    # print(f"statistics location: {model_statistics_directory}")
    
    # model parameters and model setup
    print(f"evaluation parameter: {args}")
    print(f"setting up model: {args.model}")
    model = set_model(args)
    if os.path.exists(args.ckpt):
        print(f'loading existing model:{args.ckpt}')
        model.load_state_dict(torch.load(args.ckpt))
        model.eval()
    else:
        print(f"{args.ckpt} does not exit, check checkpoint information")
        return

#----------------------------------------------- ANALYZE ID DATA------------------------------------------#

    print('---------- Processing ID Starts ------------')
    # setup directory structure
    in_model_statistics_directory = os.path.join(model_statistics_directory, f"{args.in_dataset}/")
    weight_directory = os.path.join(model_statistics_directory, f"Weights/")
    in_scales_directory = os.path.join(in_model_statistics_directory, f"in_scales/")
    in_features_directory = os.path.join(in_model_statistics_directory, f"in_features/")
    for directory in [weight_directory, in_scales_directory, in_features_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    # print(f"in stats file location: {weight_directory, in_scales_directory, in_features_directory}")
    
    # # save W: weight matrix f(x) = W^Th(x) + b
    # save_weights(args, model, weight_directory)
    # # save features
    # train_set, test_set = load_in_dataset(args)
    # labels, avg, std, maxi, median, entropy, logit = obtain_in_statistics(args, model, test_set)
    # save_in_features(in_features_directory, in_scales_directory, avg, std, maxi, median, entropy, logit, labels)

    print('---------- Processing ID Finished -------------')

#----------------------------------------------- ANALYZE OOD DATA-----------------------------------------#

    print('---------- Processing OOD Starts ------------')
    if args.in_dataset == 'ImageNet-1K':
        out_datasets = [ 'imagenet_noise'] #['SUN', 'Places', 'imagenet_dtd', 'iNaturalist'] #[ 'imagenet_noise']
    elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]: 
        out_datasets = [ 'cifar_noise'] #[ 'SVHN', 'places365', 'iSUN', 'dtd', 'LSUN', 'LSUN_resize'] #[ 'cifar_noise']

    for out_dataset in out_datasets:
        args.out_dataset = out_dataset
        print(f"processing out_dataset: {out_dataset}")

        # setup directory structure
        out_model_statistics_directory = os.path.join(model_statistics_directory, f"{args.out_dataset}")
        out_scales_directory = os.path.join(out_model_statistics_directory, f"out_scales/")
        out_features_directory = os.path.join(out_model_statistics_directory, f"out_features/")
        for directory in [out_scales_directory, out_features_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        # print(f"out stats file location: {out_scales_directory, out_features_directory}")

        # save features
        train_set, test_set = load_out_dataset(args) # train_set is dummy string.
        avg, std, maxi, median, entropy, logit = obtain_out_statistics(args, model, test_set)
        save_out_features(out_features_directory, out_scales_directory, avg, std, maxi, median, entropy, logit)

    print('---------- Processing OOD Finished ------------')

    
if __name__ == '__main__':
    args = args_parser()
    main(args)