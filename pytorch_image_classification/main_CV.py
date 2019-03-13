#!/usr/bin/env python
# coding: utf-8
print(' in main_CV.py')
import os
import time
import json
import logging
import argparse
import numpy as np
import random
import hashlib
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

from dataloader_cv import get_loader
print('loaded dataloader cv')
from utils import (str2bool, load_model, save_checkpoint, create_optimizer,
                   AverageMeter, mixup, CrossEntropyLoss)

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

    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--test_first', type=str2bool, default=True)
    parser.add_argument('--gpu', type=str, default='0')
#off    parser.add_argument('--gpu', type=str, default='-1')

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
        choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST'])
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

    parser.add_argument('--held_out', type=int, required=False)

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    config = get_config(args)

    return config


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config,
          writer):
    global global_step

    run_config = config['run_config']
    
    optim_config = config['optim_config']
    data_config = config['data_config']

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if data_config['use_mixup']:
            data, targets = mixup(data, targets, data_config['mixup_alpha'],
                                  data_config['n_classes'])

        if run_config['tensorboard_train_images']:
            if epoch == 0 and step == 0:
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
        if data_config['use_mixup']:
            _, targets = targets.max(dim=1)
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

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

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, model, criterion, test_loader, run_config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    for step, (data, targets) in enumerate(test_loader):
        if epoch == 0 and step == 0:
            h = hashlib.sha256()
            print('epoch ', epoch)
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

            # remove below

        if run_config['tensorboard_test_images']:
            if epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Test/Image', image, epoch)

        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()

        with torch.no_grad():
            outputs = model(data)
        loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        loss_meter.update(loss_, num)
        correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

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
    global global_step
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    run_config['resume'] = False
    optim_config = config['optim_config']

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

    # create output directory
    outdir = run_config['outdir']
    print('outdir: ', outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save config as json file in output directory
    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # load data loaders
    # add argument here
    held_out = run_config['held_out']
    train_loader, test_loader = get_loader(config['data_config'], held_out)

    # load model
    logger.info('Loading model...')
    model = load_model(config['model_config'])
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))
    if run_config['use_gpu']:
        model = nn.DataParallel(model)
        model.cuda()
    logger.info('Done')

    if config['data_config']['use_mixup']:
        train_criterion = CrossEntropyLoss(size_average=True)
    else:
        train_criterion = nn.CrossEntropyLoss(size_average=True)
    test_criterion = nn.CrossEntropyLoss(size_average=True)

    # create optimizer
    optim_config['steps_per_epoch'] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    start_epoch = 1

    # load pretrained weights if given
    if run_config['resume']:
        if os.path.isfile(run_config['resume']):
            print("=> loading checkpoint '{}'".format(run_config['resume']))
            checkpoint = torch.load(run_config['resume'])
            loaded_last_epoch = checkpoint['epoch']
            start_epoch = loaded_last_epoch + 1
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(run_config['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(run_config['resume']))

    if start_epoch != 1:
        for epoch in range(1, start_epoch):
            for step, (data, targets) in enumerate(train_loader):
                global_step += 1

                if optim_config['scheduler'] == 'multistep':
                    scheduler.step(epoch - 1)
                elif optim_config['scheduler'] == 'cosine':
                    scheduler.step()

    # run test before start training
    if run_config['test_first']:
        test(0, model, test_criterion, test_loader, run_config, writer)

    state = {
        'config': config,
        'state_dict': None,
        'optimizer': None,
        'epoch': 0,
        'accuracy': 0,
        'best_accuracy': 0,
        'best_epoch': 0,
    }
    for epoch in range(start_epoch, optim_config['epochs'] + 1):
        # train
        train(epoch, model, optimizer, scheduler, train_criterion,
              train_loader, config, writer)

        # test
        accuracy = test(epoch, model, test_criterion, test_loader, run_config,
                        writer)

        # update state dictionary
        state = update_state(state, epoch, accuracy, model, optimizer)
        print('outdir 2: ', outdir)
        # save model
        save_checkpoint(state, outdir)
#        save_checkpoint_epoch(state, epoch, outdir)
    
    if run_config['tensorboard']:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)

    print('completed')

if __name__ == '__main__':
    main()
