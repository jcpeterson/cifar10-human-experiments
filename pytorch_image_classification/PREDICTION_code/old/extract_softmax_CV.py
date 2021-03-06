#!/usr/bin/env python
# coding: utf-8

# example for testing this script
# python extract.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/best_model_state.pth --gpu 0 --no_output --test_only

import os
import hashlib
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

#from dataloader_extract_softmax import get_loader
# add in
from dataloader_cv_extract_softmax import get_loader
print('loaded dataloader cv extract')
from utils import (str2bool, load_model, save_checkpoint, create_optimizer,
                   AverageMeter, mixup, CrossEntropyLoss, onehot)

#from rutils_run import save_checkpoint_epoch

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
    # where to save the loss/accuracy for c10h to a csv file
    parser.add_argument('--c10h_scores_outdir', type=str, default='tmp')
    # c10h scores save interval (in epochs)
    parser.add_argument('--c10h_save_interval', type=int, default=1)
    # how much of the data to use use for test for c10h training
    parser.add_argument('--c10h_testsplit_percent', type=float, default=0.1)
    # seed for splitting the c10h data into train/test
    parser.add_argument('--c10h_datasplit_seed', type=int, default=999)
    # whether to use the cifar10 labels for the human test set (CONTROL)
    parser.add_argument('--nonhuman_control', action='store_true', default=False)
    # whether to sample from the human labels to get one-hot samples
    parser.add_argument('--c10h_sample', action='store_true', default=False)
    # whether to save to out_dir
    parser.add_argument('--no_output', action='store_true', default=False)
    # to test the loaded model and don't train
    parser.add_argument('--test_only', action='store_true', default=False)

    parser.add_argument('--cv_index', type=int, required=False)



    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    config = get_config(args)

    return config


def test(model, test_loader, run_config):

    model.eval()

    target_list = []

    output_list = []

    probs_list = []

    for step, batch_data in enumerate(test_loader):
        data, targets = batch_data
        if step == 0:
            h = hashlib.sha256()
            print('printing hash of first data row[:5]')
            d = data.cpu().numpy() # maybe this step not needed
            print(d.shape)
            d = d[0, 0, 0, :5]
            print(d.shape)
            print(d)
            d = (d.tostring())
            print(d)
            h.update(d)
            h.hexdigest()



        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()

        with torch.no_grad():
            outputs = model(data)
            sft = nn.Softmax()
            probs = sft(outputs)

        target_list.append(targets.cpu().numpy()) #var.data.numpy()
        output_list.append(outputs.cpu().numpy())
        probs_list.append(probs.cpu().numpy())
        
    return np.concatenate(target_list), np.vstack(output_list), np.vstack(probs_list)

def train(model, train_loader, run_config):
    # NB, in special version of dataloader, train shuffle turned off
    model.eval()

    target_list = []

    output_list = []

    probs_list = []
    
    for step, batch_data in enumerate(train_loader):

        data, targets = batch_data

        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()#

        with torch.no_grad():
            outputs = model(data)
            sft = nn.Softmax()
            probs = sft(outputs)

        target_list.append(targets.cpu().numpy()) #var.data.numpy()
        output_list.append(outputs.cpu().numpy())
        probs_list.append(probs.cpu().numpy())#

    return np.concatenate(target_list), np.vstack(output_list), np.vstack(probs_list)

def main():
    # parse command line argument and generate config dictionary
    config = parse_args()

    run_config = config['run_config']


    # load data loaders
    held_out = run_config['held_out']

    train_loader, test_loader = get_loader(config['data_config'], held_out)

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
    labels_test, outputs_test, probs_test = test(model, test_loader, run_config)

    labels_train, outputs_train, probs_train = train(model, train_loader, run_config)

    outdir = run_config['outdir'] # add outdir to call
    
    print('test_labels shape', labels_test.shape)
    print('train_labels shape', labels_train.shape)

    np.savez(os.path.join(str(outdir), str(run_config['resume'].split('/')[-2])) + '_test', labels=labels_test, 
                                                                                            logits=outputs_test,
                                                                                            probs=probs_test)
    np.savez(os.path.join(str(outdir), str(run_config['resume'].split('/')[-2])) + '_train', labels=labels_train, 
                                                                                             logits=outputs_train,
                                                                                             probs=probs_train)

if __name__ == '__main__':
    # main_tune()
    main()

    
    
