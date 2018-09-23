#!/usr/bin/env python
# coding: utf-8

# example for testing this script
# python extract.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/best_model_state.pth --gpu 0 --no_output --test_only

import os
import time
import json
import logging
import argparse
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader

from utils import (str2bool, load_model, save_checkpoint, create_optimizer,
                   AverageMeter, mixup, CrossEntropyLoss, onehot)

from rutils_run import save_checkpoint_epoch

from argparser import get_config

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str)
    parser.add_argument('--config', type=str)

    # model config (VGG)
    parser.add_argument('--n_channels', type=str)
    parser.add_argument('--n_layers', type=str)
    parser.add_argument('--use_bn', type=str2bool)
    #
    parser.add_argument('--base_channels', type=int)
    parser.add_argument('--block_type', type=str)
    parser.add_argument('--depth', type=int)
    # model config (ResNet-preact)
    parser.add_argument('--remove_first_relu', type=str2bool)
    parser.add_argument('--add_last_bn', type=str2bool)
    parser.add_argument('--preact_stage', type=str)
    # model config (WRN)
    parser.add_argument('--widening_factor', type=int)
    # model config (DenseNet)
    parser.add_argument('--growth_rate', type=int)
    parser.add_argument('--compression_rate', type=float)
    # model config (WRN, DenseNet)
    parser.add_argument('--drop_rate', type=float)
    # model config (PyramidNet)
    parser.add_argument('--pyramid_alpha', type=int)
    # model config (ResNeXt)
    parser.add_argument('--cardinality', type=int)
    # model config (shake-shake)
    parser.add_argument('--shake_forward', type=str2bool)
    parser.add_argument('--shake_backward', type=str2bool)
    parser.add_argument('--shake_image', type=str2bool)
    # model config (SENet)
    parser.add_argument('--se_reduction', type=int)

    parser.add_argument('--outdir', type=str, required=False)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--test_first', type=str2bool, default=True)
    parser.add_argument('--gpu', type=str, default='0') # -1 for CPU

    # TensorBoard configuration
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_train_images', action='store_true')
    parser.add_argument('--tensorboard_test_images', action='store_true')
    parser.add_argument('--tensorboard_model_params', action='store_true')

    # configuration of optimizer
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    # configuration for SGD
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--nesterov', type=str2bool)
    # configuration for learning rate scheduler
    parser.add_argument(
        '--scheduler', type=str, choices=['none', 'multistep', 'cosine'])
    # configuration for multi-step scheduler]
    parser.add_argument('--milestones', type=str)
    parser.add_argument('--lr_decay', type=float)
    # configuration for cosine-annealing scheduler]
    parser.add_argument('--lr_min', type=float, default=0)
    # configuration for Adam
    parser.add_argument('--betas', type=str)

    # configuration of data loader
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR10H'])
    parser.add_argument('--num_workers', type=int, default=7)
    # cutout configuration
    parser.add_argument('--use_cutout', action='store_true', default=False)
    parser.add_argument('--cutout_size', type=int, default=16)
    parser.add_argument('--cutout_prob', type=float, default=1)
    parser.add_argument('--cutout_inside', action='store_true', default=False)
    # random erasing configuration
    parser.add_argument(
        '--use_random_erasing', action='store_true', default=False)
    parser.add_argument('--random_erasing_prob', type=float, default=0.5)
    parser.add_argument(
        '--random_erasing_area_ratio_range', type=str, default='[0.02, 0.4]')
    parser.add_argument(
        '--random_erasing_min_aspect_ratio', type=float, default=0.3)
    parser.add_argument('--random_erasing_max_attempt', type=int, default=20)
    # mixup configuration
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', type=float, default=1)

    # previous model weights to load if any
    parser.add_argument('--resume', type=str)
    # whether to tune to human labels
    parser.add_argument('--human_tune', action='store_true', default=False)
    # whether to save to out_dir
    parser.add_argument('--no_output', action='store_true', default=False)
    # to test the loaded model and don't train
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    config = get_config(args)

    return config


def test(model, criterion, test_loader, run_config):

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()

    target_list = []

    output_list = []

    for step, batch_data in enumerate(test_loader):

        data, targets = batch_data

#        print('targets at step ', step, targets)

        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()

        with torch.no_grad():
            outputs = model(data)

#        print('outputs at step ', step, outputs)

        n_obs = data.size(0)

        # compute loss for each test set
        loss = criterion(outputs, targets)

        # save losses
        loss_meter.update(loss.item(), n_obs)

        # turn the NN probs into classifications ("predictions")
        _, preds = torch.max(outputs, dim=1)

        correct_ = preds.eq(targets).sum().item()
        correct_meter.update(correct_, 1)
 
        target_list.append(targets.cpu().numpy()) #var.data.numpy()
        output_list.append(outputs.cpu().numpy())

#        print('\n target \n', target_list[-1], '\n output \n', output_list[-1])

    accuracy = correct_meter.sum / float(len(test_loader.dataset))

    return np.concatenate(target_list), np.vstack(output_list), np.array(accuracy)

def main():
    # parse command line argument and generate config dictionary
    config = parse_args()

    run_config = config['run_config']


    # load data loaders
    train_loader, test_loader = get_loader(config['data_config'])

    # load model
    model = load_model(config['model_config'])
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    if run_config['use_gpu']:
        model = nn.DataParallel(model)
        model.cuda()

    test_criterion = nn.CrossEntropyLoss(size_average=True)

   # load pretrained weights if given
    if run_config['resume']:
        if os.path.isfile(run_config['resume']):
            print("=> loading checkpoint '{}'".format(run_config['resume']))
            checkpoint = torch.load(run_config['resume'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(run_config['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(run_config['resume']))

    # get labels
    labels, outputs, accuracy = test(model, test_criterion, 
            test_loader, 
            run_config)

    outdir = run_config['outdir'] # add outdir to call
    
    print(labels.shape)

    np.savez(os.path.join(str(outdir), str(run_config['resume'].split('/')[-2])), labels=labels, outputs=outputs, accuracy=accuracy)

if __name__ == '__main__':
    # main_tune()
    main()

    
    
