import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from configs import ModelConfig
from utils.utils import set_model, get_device, redirect_stdout_to_file
from utils.score import get_msp_score, get_energy_score, get_odin_score
from utils.display import print_measures, get_and_print_results, save_excel_sheet
from utils.datasets import load_in_dataset, load_out_dataset, load_feature_dataset, load_feature_scale_dataset, dataset_loader


def args_parser():
    parser = argparse.ArgumentParser(description='Evaluates OOD Detector',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--p', default=None, type=int, help='DICE pruning level')
    parser.add_argument('--ash_p', default=None, type=int, help='ASH pruning level')
    parser.add_argument('--threshold', default=None, type=float, help='ReAct threshold')
    parser.add_argument('--epoch', default ="100", type=str, help='which epoch to test')
    parser.add_argument('--scale_threshold', default=None, type=float, help='Scale threshold')
    parser.add_argument('--ood_scale_type', default='avg', type=str, help='true for during evaluation')
    parser.add_argument('--ood_eval_type', default='standard', type=str, help='true for during evaluation')
    parser.add_argument('--ood_eval_method', default='AdaptScale', type=str, help='[Baseline, ReAct, DICE, ASH, adaptive]')

    parser.add_argument('--embedding_dim', default = 512, type=int, help='penultimate feature dim')
    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--ood_loc', default="datasets/ood/", type=str, help='location of ood datasets')
    parser.add_argument('--out-dataset', default="iSUN-dummy", type=str, help='out-distribution dataset')
    parser.add_argument('--id_loc', default="datasets/in/", type=str, help='location of in-distribution dataset')
    
    parser.add_argument('--batch-size', default= 1024, type=int, help='mini-batch size')
    parser.add_argument('--checkpoint', default = 'model', type=str, help='checkpoint name')
    parser.add_argument('--normalize', action='store_true', help='normalize feat embeddings')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in in-dataset')
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, resnet34, densenet101]')

    parser.add_argument('--noise', type=float, default=0.0015, help='noise for odin')
    parser.add_argument('--temp', type=float, default=1.0, help='temperature: energy|odin')
    parser.add_argument('--score', default='energy', type=str, help='score options: msp|odin|energy|knn|maha')

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

    # excel template to save results.
    if args.in_dataset == 'ImageNet-1K':
        result_template_file_path = os.path.join(ModelConfig.ood_evaluation_directory, f"results_imagenet.xlsx")
    elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]: 
        result_template_file_path = os.path.join(ModelConfig.ood_evaluation_directory, f"results_cifar.xlsx")
    
    # 1. standard evaluation 2. adaptive evaluation: avg, max std, entropy, median, sobel, combined
    base_ood_directory_name = f"{args.model}/{args.in_dataset}/{args.ood_eval_method}/{args.ood_eval_type}/"
    ood_evaluation_directory = os.path.join(ModelConfig.ood_evaluation_directory, base_ood_directory_name)
    if args.ood_eval_type == 'adaptive':
        scale_based_directory = f"{args.ood_scale_type}/"
        ood_evaluation_directory = os.path.join(ood_evaluation_directory, scale_based_directory)

    # load checkpoints
    model_checkpoint_name = f"{args.checkpoint}.pt"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)
    args.ckpt = os.path.join(model_checkpoint_directory, model_checkpoint_name)

    # feature statistics name
    info = f"{args.checkpoint}_feature_stats.npy"
    args.info = os.path.join(model_checkpoint_directory, info)
    if args.p is not None:
        args.info = np.load(args.info)

    if not os.path.exists(ood_evaluation_directory):
        os.makedirs(ood_evaluation_directory)
    return ood_evaluation_directory, model_statistics_directory, result_template_file_path

def calculate_msp_score(args, model, data_set, file_name):
    device = args.device
    data_loader = dataset_loader(args, data_set, batch_size=512)
    scores = np.array([])

    # with  open(file_name, 'w') as f:
    for x,y in data_loader:
        x = x.to(device)
        batch_scores = get_msp_score(inputs=(x, y), model=model, args=args)
        # for score in batch_scores:
        #     f.write("{}\n".format(score))
        scores = np.concatenate((scores, batch_scores))
    np.save(file_name, scores)
    return scores

def calculate_odin_score(args, model, data_set, file_name):
    # note: we can't speed the evaluation using pre-computed features, because odin requires back propagation
    if args.out_dataset == 'iSUN-dummy':
        _, data_set = load_in_dataset(args)
    else:
        _, data_set = load_out_dataset(args)
    device = args.device
    data_loader = dataset_loader(args, data_set, batch_size=128)
    scores = np.array([])

    # with  open(file_name, 'w') as f:
    for x,y in data_loader:
        x = x.to(device)
        batch_scores = get_odin_score(inputs=x, model=model, args=args)
        # for score in batch_scores:
        #     f.write("{}\n".format(score))
        scores = np.concatenate((scores, batch_scores))
    np.save(file_name, scores)
    return scores

def calculate_energy_score(args, model, data_set, file_name):
    device = args.device
    data_loader = dataset_loader(args, data_set, batch_size=512)
    scores = np.array([])

    # with  open(file_name, 'w') as f:
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        batch_scores = get_energy_score(inputs=(x, y), model=model, args=args)
        # for score in batch_scores:
        #     f.write("{}\n".format(score))
        scores = np.concatenate((scores, batch_scores))
    np.save(file_name, scores)
    return scores

def get_score(args, model, test_set, file_name):
    if args.score == 'msp':
        return calculate_msp_score(args, model, test_set, file_name)
    elif args.score == 'odin':
        return calculate_odin_score(args, model, test_set, file_name)
    elif args.score == 'energy':
        return calculate_energy_score(args, model, test_set, file_name)
    
def get_in_feature_path(args, model_statistics_directory):
    in_scales_directory = os.path.join(model_statistics_directory, f"{args.in_dataset}/in_scales/")
    in_features_directory = os.path.join(model_statistics_directory, f"{args.in_dataset}/in_features/")
    in_features_path = in_features_directory + 'avg.npy'
    return in_features_path, in_scales_directory

def get_out_feature_path(args, model_statistics_directory):
    out_scales_directory = os.path.join(model_statistics_directory, f"{args.out_dataset}/out_scales/")
    out_features_directory = os.path.join(model_statistics_directory, f"{args.out_dataset}/out_features/")
    out_features_path = out_features_directory + 'avg.npy'
    return out_features_path, out_scales_directory

def get_scale_path(args, scale_directory):
        if args.ood_scale_type == 'avg':
            scale_path = scale_directory + 'avg.npy'
        elif args.ood_scale_type == 'std':
            scale_path = scale_directory + 'std.npy'
        elif args.ood_scale_type == 'max':
            scale_path = scale_directory + 'max.npy'
        elif args.ood_scale_type == 'median':
            scale_path = scale_directory + 'median.npy'
        elif args.ood_scale_type == 'entropy':
            scale_path = scale_directory + 'entropy.npy'
        elif args.ood_scale_type == 'feature_entropy':
            scale_path = scale_directory + 'feature_entropy.npy'
        return scale_path

def load_eval_dataset(args, feature_path, scale_directory, score_file_name):
    
    if args.ood_eval_type == 'standard':
        test_set = load_feature_dataset(feature_path) 
    else:
        # scale_path = get_scale_path(args, scale_directory)
        # score_path = score_file_name.replace(f"adaptive/{args.ood_scale_type}","standard")
        # test_set = load_scale_score_dataset(scale_path, score_path)
        scale_path = feature_path.replace("avg.npy",f"{args.ood_scale_type}"+".npy")
        test_set = load_feature_scale_dataset(feature_path, scale_path)
    return test_set


def main(args):

    # setting up result directory
    ood_evaluation_directory, model_statistics_directory, result_template_file_path = setup_directory(args)
    metric_log_writer = redirect_stdout_to_file(ood_evaluation_directory, f"{args.score}_metric.log")
    in_score_file_name = os.path.join(ood_evaluation_directory, f"{args.model}_{args.in_dataset}_in_{args.score}_score.npy")
    
    # evaluation parameter and model setup
    print(f"evaluation parameter: {args}")
    print(f"setting up model: {args.model}")
    model = set_model(args)
    if os.path.exists(args.ckpt):
        print(f'loading existing model:{args.ckpt}')
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model.to(args.device)
        model.eval()
    else:
        print(f"{args.ckpt} does not exit, check checkpoint information")
        return

    print('---------- Processing ID Starts ------------')

    # load in-datasets
    in_feature_path, in_scale_directory = get_in_feature_path(args, model_statistics_directory)
    test_set = load_eval_dataset(args, in_feature_path, in_scale_directory, in_score_file_name)
    in_scores = get_score(args, model, test_set, in_score_file_name)
    print(f"sample id {args.score} scores for {args.in_dataset}: {in_scores[:5]}")

    print('---------- Processing ID Finished ----------')

    print('---------- Processing OOD Starts ------------')

    if args.in_dataset == 'ImageNet-1K':
        out_datasets = ['SUN', 'Places', 'imagenet_dtd', 'iNaturalist']
    elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]: 
        # out_datasets = ['CIFAR-100'] # simulate Near-OOD for CIFAR-10
        out_datasets = [ 'SVHN', 'places365', 'iSUN', 'dtd', 'LSUN', 'LSUN_resize']

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
       
        args.out_dataset = out_dataset
        print(f"processing out_dataset: {out_dataset}")
        out_score_file_name = os.path.join(ood_evaluation_directory, f"{args.model}_{args.out_dataset}_out_{args.score}_score.npy")

        out_feature_path, out_scale_directory = get_out_feature_path(args, model_statistics_directory)
        test_set = load_eval_dataset(args, out_feature_path, out_scale_directory, out_score_file_name)

        out_scores = get_score(args, model, test_set, out_score_file_name)
        print(f"sample ood {args.score} scores for {out_dataset}: {out_scores[:5]}")

        get_and_print_results(args, in_scores, out_scores, auroc_list, aupr_list, fpr_list)
    
    print("AVG")
    print_measures(None, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list))

    auroc_mean, aupr_mean, fpr_mean = np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list)
    fpr_list.append(fpr_mean)
    aupr_list.append(aupr_mean)
    auroc_list.append(auroc_mean)

    result_file_path = os.path.join(ood_evaluation_directory, f"results.xlsx")
    hyper_parameter = f"dice_p={args.p},ash_p={args.ash_p},react_th={args.threshold},scale_th={args.scale_threshold},type={args.ood_scale_type}"
    save_excel_sheet(out_datasets, fpr_list, auroc_list, hyper_parameter, result_file_path, result_template_file_path)

    print('---------- Processing OOD Finished ----------')


    metric_log_writer.close()

    
if __name__ == '__main__':
    args = args_parser()
    # evaluate ood detection
    main(args)