#!/usr/bin/env python
# coding: utf-8

# example for testing this script
# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/model_state_160.pth --gpu 0 --no_output --test_only --c10h_sample

# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/model_state_160.pth --gpu 0 --no_output --c10h_sample --base_lr 0.01
# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/model_state_160.pth --gpu 0 --no_output --base_lr 0.01

# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch shake_shake --config tmp_reference_model/config.json --resume tmp_reference_model/model_best_state.pth --gpu 0 --no_output --base_lr 0.01
# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch shake_shake --config tmp_reference_model/config.json --resume tmp_reference_model/model_best_state.pth --gpu 0 --no_output --base_lr 0.01 --nonhuman_control


import os
import time
import json
import logging
import argparse
import numpy as np
import random

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

# from dataloader import get_loader
from dataloader_c10h import get_loader

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
    # 
    parser.add_argument('--c10h_datasplit_seed', type=int, default=999)
    # whether to use the cifar10 labels for the human test set (CONTROL)
    parser.add_argument('--nonhuman_control', action='store_true', default=False)
    # whether to sample from the human labels to get one-hot samples
    parser.add_argument('--c10h_sample', action='store_true', default=False)
    # whether to save to out_dir
    parser.add_argument('--no_output', action='store_true', default=False)
    # to test the loaded model and don't train
    parser.add_argument('--test_only', action='store_true', default=False)

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    config = get_config(args)

    return config


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config,
          writer, human_tune=False):
    global global_step

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    nonhuman_control = config['run_config']['nonhuman_control']

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, batch_data in enumerate(train_loader):
        global_step += 1

        if human_tune:
            if nonhuman_control:
                # `targets` are now `c10h_c10_targets`
                data, _, targets = batch_data
                targets = onehot(targets, 10)
            else:
                data, targets, _ = batch_data                
        else:
            data, targets = batch_data

        if data_config['use_mixup']:
            data, targets = mixup(data, targets, data_config['mixup_alpha'],
                                  data_config['n_classes'])

        if run_config['tensorboard_train_images']:
            if step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Train/Image', image, epoch)

        if optim_config['scheduler'] == 'multistep':
            scheduler.step(epoch - 1)
        elif optim_config['scheduler'] == 'cosine':
            scheduler.step()

        if run_config['tensorboard']:
            if optim_config['scheduler'] != 'none':
                lr = scheduler.get_lr()[0]
            else:
                lr = optim_config['base_lr']
            writer.add_scalar('Train/LearningRate', lr, global_step)

        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        if human_tune or data_config['use_mixup']:
            _, targets = targets.max(dim=1)
        correct_ = preds.eq(targets).sum().item()
        n_obs = data.size(0)

        accuracy = correct_ / float(n_obs)

        loss_meter.update(loss_, n_obs)
        accuracy_meter.update(accuracy, n_obs)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if not human_tune:
            if step % 100 == 0:
                logger.info('Epoch {} Step {}/{} '
                            'Loss {:.4f} ({:.4f}) '
                            'Accuracy {:.4f} ({:.4f})'.format(
                                epoch,
                                step,
                                len(train_loader),
                                loss_meter.val,
                                loss_meter.avg,
                                accuracy_meter.val,
                                accuracy_meter.avg,
                            ))
    if human_tune:
        logger.info('Train Epoch {} Loss {:.4f} (acc: {:.4f})'.format(
                        epoch,
                        loss_meter.avg,
                        accuracy_meter.avg,
                    ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)

def test(epoch, model, criterion, test_loaders, run_config, writer, 
        human_tune=False):

    if human_tune:
        test_loader, v4_loader, v6_loader = test_loaders
    else:
        test_loader = test_loaders

    logger.info('TEST {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    if human_tune:
        c10h_c10_loss_meter = AverageMeter()
        c10h_c10_correct_meter = AverageMeter()
        v4_loss_meter = AverageMeter()
        v4_correct_meter = AverageMeter()
        v6_loss_meter = AverageMeter()
        v6_correct_meter = AverageMeter()
    start = time.time()

    for step, batch_data in enumerate(test_loader):

        if human_tune:
            data, targets, c10h_c10_targets = batch_data
            c10h_c10_targets = onehot(c10h_c10_targets, 10)
        else:
            data, targets = batch_data

        if run_config['tensorboard_test_images']:
            if epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Test/Image', image, epoch)

        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()
            if human_tune: 
                c10h_c10_targets = c10h_c10_targets.cuda()

        with torch.no_grad():
            outputs = model(data)

        n_obs = data.size(0)

        # compute loss for each test set
        loss = criterion(outputs, targets)
        if human_tune:
            c10h_c10_loss = criterion(outputs, c10h_c10_targets)

        # save losses
        loss_meter.update(loss.item(), n_obs)
        if human_tune:
            c10h_c10_loss_meter.update(c10h_c10_loss.item(), n_obs)

        # turn the NN probs into classifications ("predictions")
        _, preds = torch.max(outputs, dim=1)

        if human_tune: 
            _, targets = targets.max(dim=1)
            _, c10h_c10_targets = c10h_c10_targets.max(dim=1)

        correct_ = preds.eq(targets).sum().item()
        correct_meter.update(correct_, 1)

        if human_tune:
            c10h_c10_correct_ = preds.eq(c10h_c10_targets).sum().item()
            c10h_c10_correct_meter.update(c10h_c10_correct_, 1)

    # TEST v4/v6 CIFAR 10.1
    if human_tune:
        for step, (data, targets) in enumerate(v4_loader):
            targets = onehot(targets, 10)
            if run_config['use_gpu']:
                data = data.cuda()
                targets = targets.cuda()
            with torch.no_grad(): outputs = model(data)
            n_obs = data.size(0)
            v4_loss = criterion(outputs, targets)
            v4_loss_meter.update(v4_loss.item(), n_obs)
            _, preds = torch.max(outputs, dim=1)
            _, targets = targets.max(dim=1)
            v4_correct_ = preds.eq(targets).sum().item()
            v4_correct_meter.update(v4_correct_, 1)

        for step, (data, targets) in enumerate(v6_loader):
            targets = onehot(targets, 10)
            if run_config['use_gpu']:
                data = data.cuda()
                targets = targets.cuda()
            with torch.no_grad(): outputs = model(data)
            n_obs = data.size(0)
            v6_loss = criterion(outputs, targets)
            v6_loss_meter.update(v6_loss.item(), n_obs)
            _, preds = torch.max(outputs, dim=1)
            _, targets = targets.max(dim=1)
            v6_correct_ = preds.eq(targets).sum().item()
            v6_correct_meter.update(v6_correct_, 1)

    accuracy = correct_meter.sum / float(len(test_loader.dataset))
    if human_tune:
        c10h_c10_accuracy = \
            c10h_c10_correct_meter.sum / float(len(test_loader.dataset))
        v4_accuracy = v4_correct_meter.sum / float(len(v4_loader.dataset))
        v6_accuracy = v6_correct_meter.sum / float(len(v6_loader.dataset))

    if human_tune:
        logger.info('- epoch {}, c10h: {:.4f} (acc: {:.4f}) | c10h_c10: {:.4f} (acc: {:.4f})'.format(
            epoch, loss_meter.avg, accuracy, c10h_c10_loss_meter.avg, c10h_c10_accuracy))
        logger.info('-       {}    v4: {:.4f} (acc: {:.4f}) |       v6: {:.4f} (acc: {:.4f})'.format(
            epoch, v4_loss_meter.avg, v4_accuracy, v6_loss_meter.avg, v6_accuracy))
    else:
        logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    if run_config['tensorboard_model_params']:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    return accuracy


def update_state(state, epoch, accuracy, model, optimizer):
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['epoch'] = epoch
    state['accuracy'] = accuracy

    # update best accuracy
    if accuracy > state['best_accuracy']:
        state['best_accuracy'] = accuracy
        state['best_epoch'] = epoch

    return state


def main():
    # parse command line argument and generate config dictionary
    config = parse_args()

    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']

    human_tune = run_config['human_tune']

    # TensorBoard SummaryWriter
    if run_config['tensorboard']:
        writer = SummaryWriter()
    else:
        writer = None

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not run_config['no_output']:
        # create output directory
        outdir = run_config['outdir']
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save config as json file in output directory
        outpath = os.path.join(outdir, 'config.json')
        with open(outpath, 'w') as fout:
            json.dump(config, fout, indent=2)

    # load data loaders
    if human_tune:
        train_loader, test_loader, v4_loader, v6_loader = \
            get_loader(config['data_config'])
    else:
        train_loader, test_loader = get_loader(config['data_config'])

    # load model
    logger.info('Loading model...')
    model = load_model(config['model_config'])
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))
    if run_config['use_gpu']:
        model = nn.DataParallel(model)
        model.cuda()
    logger.info('Done')

    if human_tune or config['data_config']['use_mixup']:
        train_criterion = CrossEntropyLoss(size_average=True)
        test_criterion = CrossEntropyLoss(size_average=True)
    else:
        train_criterion = nn.CrossEntropyLoss(size_average=True)
        test_criterion = nn.CrossEntropyLoss(size_average=True)

    # create optimizer
    optim_config['steps_per_epoch'] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    # load pretrained weights if given
    if run_config['resume']:
        if os.path.isfile(run_config['resume']):
            print("=> loading checkpoint '{}'".format(run_config['resume']))
            checkpoint = torch.load(run_config['resume'])
            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(run_config['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(run_config['resume']))

    # run test before we start training
    if run_config['resume']: 
        print('Test accuracy of pretrained model --------------------')
    else:
        print('Test accuracy of untrained model ---------------------')
    if run_config['test_first']:
        if human_tune:
            test(0, model, test_criterion, 
                (test_loader, v4_loader, v6_loader), 
                run_config, writer,
                human_tune=human_tune)
        else:           
            test(0, model, test_criterion, test_loader,
                run_config, writer, human_tune=human_tune)

    if run_config['test_only']: exit()

    state = {
        'config': config,
        'state_dict': None,
        'optimizer': None,
        'epoch': 0,
        'accuracy': 0,
        'best_accuracy': 0,
        'best_epoch': 0,
    }

    for epoch in range(1, optim_config['epochs'] + 1):
        # train
        train(epoch, model, optimizer, scheduler, train_criterion,
              train_loader, config, writer, human_tune=human_tune)

        # test
        if human_tune:
            accuracy = test(epoch, model, test_criterion, 
                (test_loader, v4_loader, v6_loader), 
                run_config, writer,
                human_tune=human_tune)
        else:           
            accuracy = test(epoch, model, test_criterion, test_loader,
                            run_config, writer, human_tune=human_tune)
        # accuracy = test(epoch, model, test_criterion, test_loader, run_config,
        #                 writer, human_tune=human_tune)

        # update state dictionary
        state = update_state(state, epoch, accuracy, model, optimizer)
        
        if not run_config['no_output']:
            # save model
            save_checkpoint(state, outdir)

    if not run_config['no_output'] and run_config['tensorboard']:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)

if __name__ == '__main__':
    main()
