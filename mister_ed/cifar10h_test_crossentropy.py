# EXTERNAL LIBRARY IMPORTS
import numpy as np 
import scipy 

import torch # Need torch version 0.3 or 0.4
import torch.nn as nn 
import torch.optim as optim 
assert torch.__version__[:3] in ['0.3', '0.4']

from pytorch_image_classification_dataloader_c10h import get_loader

# use_gpu = torch.cuda.is_available()
# print(use_gpu)

use_gpu = False

# MISTER ED SPECIFIC IMPORT BLOCK
# (here we do things so relative imports work )
# Universal import block 
# Block to get the relative imports working 

import os, sys, re
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
import config
import prebuilt_loss_functions as plf
import loss_functions as lf 
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import adversarial_attacks as aa
import adversarial_training as advtrain
import adversarial_evaluation as adveval
import utils.checkpoints as checkpoints
import adversarial_perturbations as ap 
import adversarial_attacks_refactor as aar
import spatial_transformers as st

### START pytorch_image_classification imports
import time, random, json, logging, argparse, csv
from pytorch_image_classification_dataloader_c10h import get_loader
from pytorch_image_classification_utils import (str2bool, load_model, save_checkpoint, create_optimizer,
                                                AverageMeter, mixup, CrossEntropyLoss, onehot)
# from rutils_run import save_checkpoint_epoch
from pytorch_image_classification_argparser import get_config

sys.argv = ['']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--config', type=str, default='tmp_reference_model/resnet_basic_110_config.json')
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
    parser.add_argument('--batch_size', type=int, default=16)
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
        default='CIFAR10H',
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
    parser.add_argument('--c10h_save_interval', type=str, default='1') # changed from int
    # how much of the data to use use for test for c10h training
    parser.add_argument('--c10h_testsplit_percent', type=float, default=0.1)
    # seed for splitting the c10h data into train/test
    parser.add_argument('--c10h_datasplit_seed', type=int, default=999)
    # whether to use the cifar10 labels for the human test set (CONTROL)
    parser.add_argument('--nonhuman_control', type=str2bool, default=False)
    # whether to sample from the human labels to get one-hot samples
    parser.add_argument('--c10h_sample', action='store_true', default=False)
    # whether to save to out_dir
    parser.add_argument('--no_output', action='store_true', default=False)
    # to test the loaded model and don't train
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()
    # if not is_tensorboard_available:
    args.tensorboard = False

    config = get_config(args)

    return config

config = parse_args()

run_config = config['run_config']


def load_our_model(config, weights_path):
        
    our_model = load_model(config['model_config'])
    
    # load pretrained weights if given
    if os.path.isfile(weights_path):
        print("=> loading checkpoint '{}'".format(weights_path))

        # Resolve CPU/GPU stuff
        if use_gpu:
            map_location = None
        else:
            map_location= (lambda s, l: s)

        checkpoint = torch.load(weights_path,
                                map_location=map_location)

        correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in
                              checkpoint['state_dict'].items()}

        our_model.load_state_dict(correct_state_dict)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(weights_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(weights_path))
        
    return our_model

# model = load_our_model(config, run_config['resume'])

cifar_valset = cifar_loader.load_cifar_data('val', batch_size=32)

_, normalizer = cifar_loader.load_pretrained_cifar_resnet(flavor=32, use_gpu=use_gpu,
                                                          return_normalizer=True)

run_config['resume'] = 'tmp_reference_model/model_state_resnet_basic_110_con_True_lr_001_seed_0.pth'
#run_config['resume'] = 'tmp_reference_model/model_best_state.pth'
model_gt = load_our_model(config, run_config['resume'])

run_config['resume'] = 'tmp_reference_model/model_state_resnet_basic_110_con_False_lr_001_seed_0.pth'
model_human = load_our_model(config, run_config['resume'])

if use_gpu:
    model_gt.cuda()
    model_human.cuda()
    
models = [model_gt, model_human]



human_before_loss = []
gt_before_loss = []
human_after_loss = []
gt_after_loss = []

human_before_acc = []
gt_before_acc = []
human_after_acc = []
gt_after_acc = []

# cifar_valset = cifar_loader.load_cifar_data('val', batch_size=500)

train_loader, test_loader, _50k_loader, v4_loader, v6_loader = \
    get_loader(config['data_config'])


delta_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf', 
                                                 'lp_bound': 8.0 / 255,
                                                 'use_gpu': use_gpu}) 
# for examples, labels in iter(cifar_valset):
for examples, labels in iter(v4_loader):
    if use_gpu:
        examples = examples.cuda()
        labels = labels.cuda() 
    for i, m in enumerate(models):

        attack_loss = plf.VanillaXentropy(m, normalizer)
        fgsm_attack_object = aar.FGSM(m, normalizer, delta_threat, 
                                      attack_loss, use_gpu=use_gpu)

        # verbose prints out accuracy
        perturbation_out, loss_before, loss_after, acc_before, acc_after = \
            fgsm_attack_object.attack_josh(examples, labels, verbose=True)

#         print(loss_before, loss_after, acc_before, acc_after)
        
        if i == 1:
            human_before_loss.append(loss_before)
            human_after_loss.append(loss_after)
            human_before_acc.append(acc_before)
            human_after_acc.append(acc_after)
            print('\nCurrent Human Totals:')
            print(np.mean(human_before_loss), 
                  np.mean(human_after_loss), 
                  np.mean(human_before_acc), 
                  np.mean(human_after_acc))
        else:
            gt_before_loss.append(loss_before)
            gt_after_loss.append(loss_after)
            gt_before_acc.append(acc_before)
            gt_after_acc.append(acc_after)
            print('\nCurrent GT Totals:')
            print(np.mean(gt_before_loss), 
                  np.mean(gt_after_loss), 
                  np.mean(gt_before_acc), 
                  np.mean(gt_after_acc))