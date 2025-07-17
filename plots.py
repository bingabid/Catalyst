import os
import argparse
import numpy as np
import seaborn as sns
from configs import ModelConfig
import matplotlib.pyplot as plt

def args_parser():
    parser = argparse.ArgumentParser(description='Analyze Distribution',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--out-dataset', default="iSUN", type=str, help='out-distribution dataset')
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, resnet34, densenet101]')
    args = parser.parse_args()
    return args

def plot_raw(args, delta_max, delta_mean, delta_std, model_plot_directory):
    fname =  os.path.join(model_plot_directory, f"raw_stats_{args.model}_{args.in_dataset}_{args.out_dataset}") 
    plt.figure(figsize=(10, 4))
    plt.plot(delta_max, label='max')
    plt.plot(delta_std, label='std')
    plt.plot(delta_mean, label='mean')

    plt.title('')
    plt.xlabel('units')
    plt.ylabel('activaton gap')
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.savefig(fname)
    plt.close()

def plot_indexed(args, delta_max, delta_mean, delta_std, model_plot_directory):
    fname =  os.path.join(model_plot_directory, f"indexed_stats_{args.model}_{args.in_dataset}_{args.out_dataset}") 
    idx = np.argsort(delta_max)
    delta_max = delta_max[idx]
    delta_std = delta_std[idx]
    delta_mean = delta_mean[idx]
    delta_mean_std = delta_mean[idx] + delta_std[idx]

    plt.figure(figsize=(10, 4))
    plt.plot(delta_max, label='max')
    plt.plot(delta_mean_std, label='avg+std')
    plt.plot(delta_mean, label='avg')

    plt.title('')
    plt.xlabel('units')
    plt.ylabel('activaton gap')
    plt.legend(loc = 'best')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def plot_sorted(args, delta_max, delta_mean, delta_std, model_plot_directory):
    fname =  os.path.join(model_plot_directory, f"sorted_stats_{args.model}_{args.in_dataset}_{args.out_dataset}") 

    delta_max.sort()
    delta_mean_std = delta_mean + delta_std
    delta_mean_std.sort()
    delta_mean.sort()

    plt.figure(figsize=(10, 4))
    plt.plot(delta_max, label='max')
    plt.plot(delta_mean_std, label='avg+std')
    plt.plot(delta_mean, label='avg')

    plt.title('')
    plt.xlabel('units')
    plt.ylabel('activaton gap')
    plt.legend(loc = 'best')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_density(args, id_avg_score, ood_avg_score, id_max_score, ood_max_score, model_plot_directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fname = os.path.join(model_plot_directory, f"density_{args.model}_{args.in_dataset}_{args.out_dataset}") 
    # avg score
    sns.kdeplot(id_avg_score, label='in', fill=True, ax=axes[0], color='green')
    sns.kdeplot(ood_avg_score, label='ood', fill=True, ax=axes[0], color='red')
    axes[0].set_title('avg')
    axes[0].set_xlabel('scores')
    axes[0].set_ylabel('density')
    axes[0].legend()
    axes[0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # max score
    sns.kdeplot(id_max_score, label='in', fill=True, ax=axes[1], color='green')
    sns.kdeplot(ood_max_score, label='ood', fill=True, ax=axes[1], color='red')
    axes[1].set_title('max')
    axes[1].set_xlabel('scores')
    axes[1].set_ylabel('density')
    axes[1].legend()
    axes[1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def main(args):

    # setup directory path
    base_directory_name = f"{args.model}/{args.in_dataset}/"

    #result/score directory
    model_result_directory = os.path.join(ModelConfig.ood_evaluation_directory, base_directory_name)
    model_result_directory = os.path.join(model_result_directory, f"DomAct/")

    # plot directory
    model_plot_directory = os.path.join(ModelConfig.model_plot_directory, base_directory_name)
    if not os.path.exists( model_plot_directory):
        os.makedirs(model_plot_directory)
    
    #---------------------------------------------STATISTICS------------------------------------------------#
    # statistics directory
    model_statistics_directory = os.path.join(ModelConfig.model_statistics_directory, base_directory_name)
    in_model_statistics_directory = os.path.join(model_statistics_directory, f"{args.in_dataset}")
    in_statistics_file_name = os.path.join(in_model_statistics_directory, f"in_{args.in_dataset}")

    # locate id statistics files
    id_std_fname = in_statistics_file_name + '_std.npy'
    id_max_fname = in_statistics_file_name + '_maxi.npy'
    id_mean_fname = in_statistics_file_name + '_mean.npy'
    id_labels_fname = in_statistics_file_name + '_labels.npy'

    # print(id_std_fname, id_max_fname, id_mean_fname, id_labels_fname)

    # load id statistics files
    id_std = np.load(id_std_fname)
    id_max = np.load(id_max_fname)
    id_mean = np.load(id_mean_fname)
    id_labels = np.load(id_labels_fname)

    # print sample id statistics values
    # idx = 10
    # print(f"max:  {id_max[:1, idx:idx + 5]}\nmean: {id_mean[:1, idx:idx + 5]}\nstd:  {id_std[:1,  idx:idx + 5]}")
    

    if args.in_dataset == 'ImageNet-1K':
        out_datasets = ['SUN', 'Places', 'imagenet_dtd', 'iNaturalist']
    elif args.in_dataset == 'mnist':
        out_datasets = ['omniglot']
    elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]: 
        out_datasets = [ 'SVHN', 'places365', 'iSUN', 'dtd', 'LSUN', 'LSUN_resize']

    #plot statistics
    for out_dataset in out_datasets:
        args.out_dataset = out_dataset
        # print(f"ploting: {args.in_dataset} Vs {args.out_dataset}")
        out_model_statistics_directory = os.path.join(model_statistics_directory, f"{args.out_dataset}")
        out_statistics_file_name = os.path.join(out_model_statistics_directory, f"out_{args.out_dataset}")
        # print(out_statistics_file_name)

        # locate ood statistics files
        ood_std_fname = out_statistics_file_name + '_std.npy'
        ood_max_fname = out_statistics_file_name + '_maxi.npy'
        ood_mean_fname = out_statistics_file_name + '_mean.npy'
        # print(ood_max_fname, ood_std_fname, ood_mean_fname)

        # load id statistics files
        ood_std = np.load(ood_std_fname)
        ood_max = np.load(ood_max_fname)
        ood_mean = np.load(ood_mean_fname)

        # print sample ood statistics values
        # idx = 10
        # print(f"max:  {ood_max[:1, idx:idx + 5]}\nmean: {ood_mean[:1, idx:idx + 5]}\nstd:  {ood_std[:1,  idx:idx + 5]}")

        # calculate delta for mean statistics
        expected_id_mean = id_mean.mean(0)
        expected_ood_mean = ood_mean.mean(0)
        delta_mean = expected_id_mean - expected_ood_mean

        # calculate delta for std statistics
        expected_id_std = id_std.mean(0)
        expected_ood_std = ood_std.mean(0)
        delta_std = expected_id_std - expected_ood_std

        # calculate delta for max statistics
        expected_id_mean = id_max.mean(0)
        expected_ood_mean = ood_max.mean(0)
        delta_max = expected_id_mean - expected_ood_mean

        # print(f"dealta_mean: {delta_mean.mean()}\navg delta_std: {delta_std.mean()}\navg delta_max: {delta_max.mean()}")

        #plots stastistics gap
        plot_raw(args,delta_max, delta_mean, delta_std, model_plot_directory)
        plot_indexed(args,delta_max, delta_mean, delta_std, model_plot_directory)
        plot_sorted(args,delta_max, delta_mean, delta_std, model_plot_directory)

    #---------------------------------------------DENSITY------------------------------------------------#
    id_avg_score_fname =  os.path.join(model_result_directory, f"avg/{args.model}_{args.in_dataset}_in_energy_score.txt")
    id_avg_score = np.loadtxt(id_avg_score_fname)
    
    id_max_score_fname =  os.path.join(model_result_directory, f"max/{args.model}_{args.in_dataset}_in_energy_score.txt")
    id_max_score = np.loadtxt(id_max_score_fname)
    
    # plot density
    for out_dataset in out_datasets:
        args.out_dataset = out_dataset
        # print(f"density: {args.in_dataset} Vs {args.out_dataset}")
        ood_avg_score_fname =  os.path.join(model_result_directory, f"avg/{args.model}_{args.out_dataset}_out_energy_score.txt")
        ood_avg_score = np.loadtxt(ood_avg_score_fname)
        ood_max_score_fname =  os.path.join(model_result_directory, f"max/{args.model}_{args.out_dataset}_out_energy_score.txt")
        ood_max_score = np.loadtxt(ood_max_score_fname)
        
        plot_density(args, id_avg_score, ood_avg_score, id_max_score, ood_max_score, model_plot_directory)


if __name__ == '__main__':
    args = args_parser()
    # print(args)
    main(args)