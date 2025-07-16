import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

# def ash_s(x, p=90):
#     s1 = x.sum(dim = 1)
#     t = torch.quantile(x, p/100.0, dim=1, keepdim=True)
#     x = torch.where(x > t, x, torch.zeros_like(x))
#     s2 = x.sum(dim = 1)
#     scale = s1/s2
#     x = x * torch.exp(scale[:, None])
#     return x

def ash_s(x, p):
    # x := [bt, d]
    batch_size, c = x.shape

    s1 = x.sum(dim=1) 
    k = c - int(np.round(c * p / 100.0))
    v, i = torch.topk(x, k, dim=1)
    x_pruned = torch.zeros_like(x)
    x_pruned.scatter_(dim=1, index=i, src=v)

    s2 = x_pruned.sum(dim=1)
    scale = s1 / (s2 ) #+ 1e-8)
    x_sharpened = x_pruned * torch.exp(scale[:, None])

    return x_sharpened

def get_features(x, dim_in, args):
    # Rectified Activation
    if args.threshold is not None:
        x = x.clip(max = args.threshold) 
    # ASH
    if args.ash_p is not None:
        x = x.view(-1, dim_in)
        x = ash_s(x, args.ash_p)
    return x

def get_logits(inputs, model, args, logits=None):
    if logits is None:
        model.eval()
        with torch.inference_mode():
            if args.in_dataset == 'ImageNet-1K':
                if args.model == 'mobilenetv2_imagenet':
                    logits = model.classifier(inputs)
                else:
                    logits = model.fc(inputs)
            elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]:
                logits = model.output_layer(inputs) 
                # if args.model == 'densenet101':
                #     logits = model.fc(inputs)
                # else:
                #     logits = model.output_layer(inputs)     
    return  logits

def get_scales(x, args):
    if args.scale_threshold is not None:
        s = torch.clamp(x, min=0.0, max=args.scale_threshold)
    s = torch.mean(s, dim=1)
    return s

def get_msp_score(inputs, model, args, logits=None):
    x, y = inputs
    scales = 1.0
    if args.ood_eval_type == 'adaptive':
        scales = get_scales(y, args)
        if args.ood_scale_type == "entropy":
            scales = 1.0/scales
        scales = scales.data.cpu().numpy()
    x = get_features(x, model.dim_in, args)
    logits = get_logits(x, model, args)
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    # scaling the scores
    scores = scores * scales
    return scores

def get_energy_score(inputs, model, args, logits=None):
    x, y = inputs
    # if args.ood_eval_type == 'adaptive':
    #     if args.ood_scale_type in ['entropy', 'feature_entropy']:
    #         scores = x.data.cpu() / y.data.cpu()
    #     else:
    #         scores = x.data.cpu() * y.data.cpu()
    #     return scores.numpy()
    scales = 1.0
    if args.ood_eval_type == 'adaptive':
        scales = get_scales(y, args)
        if args.ood_scale_type == "entropy":
            scales = 1.0/scales
        scales = scales.data.cpu().numpy()
        # scales = np.square(scales)

    x = get_features(x, model.dim_in, args)
    logits = get_logits(x, model, args)
    # temperature scaling
    logits = logits / args.temp 
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    # scaling the scores
    scores = scores * scales
    return scores

def get_odin_score(inputs, model, args):
    # calculating the perturbation we need to add, i.e the sign of gradient of cross entropy loss w.r.t. input using simple FGSM attack perturbation
    epsilon = args.noise
    temperature = args.temp
    criterion = nn.CrossEntropyLoss()

    # get predicted labels
    inputs.requires_grad = True
    outputs = model(inputs)
    labels = torch.argmax(outputs, dim = 1).to(inputs.device)

    # using temperature scaling
    outputs = outputs / temperature
    #back propagate loss
    loss = criterion(outputs, labels)
    loss.backward()

    # get pertubed image = image - noise*grad.sign
    gradient =  inputs.grad.data.sign()
    inputs = torch.add(inputs.data,  -epsilon*gradient)

    #get new softmax score
    model.eval()
    with torch.inference_mode():
        outputs = model(inputs)
    outputs = outputs / temperature
    scores, index = torch.max(torch.softmax(outputs, dim = 1), dim = 1)
    scores = scores.data.cpu().numpy()
    return scores