#!/usr/bin/env python
# coding: utf-8

# example for testing this script
# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/model_state_160.pth --gpu 0 --no_output --test_only --c10h_sample

# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/model_state_160.pth --gpu 0 --no_output --c10h_sample --base_lr 0.01
# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch vgg --config tmp_reference_model/config.json --resume tmp_reference_model/model_state_160.pth --gpu 0 --no_output --base_lr 0.01

# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch shake_shake --config tmp_reference_model/config.json --resume tmp_reference_model/model_best_state.pth --gpu 0 --no_output --base_lr 0.01
# python tune_with_cifar10h.py --human_tune --dataset CIFAR10H --arch shake_shake --config tmp_reference_model/config.json --resume tmp_reference_model/model_best_state.pth --gpu 0 --no_output --base_lr 0.01 --nonhuman_control


import os, time, random, json, logging, argparse, csv
import numpy as np

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

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    config = get_config(args)

    return config


def test(epoch, model, criterion, test_loaders, run_config, writer, 
        human_tune=False):

    if human_tune:
        train_loader, \
        test_loader,  \
        _50k_loader,  \
        v4_loader,    \
        v6_loader = test_loaders
    else:
        test_loader = test_loaders

    logger.info('TEST {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    if human_tune:
        c10h_val_c10_loss_meter = AverageMeter()
        c10h_val_c10_correct_meter = AverageMeter()
        c10h_train_c10_loss_meter = AverageMeter()
        c10h_train_c10_correct_meter = AverageMeter()
        train_loss_meter = AverageMeter()
        train_correct_meter = AverageMeter()
        _50k_loss_meter = AverageMeter()
        _50k_correct_meter = AverageMeter()
        v4_loss_meter = AverageMeter()
        v4_correct_meter = AverageMeter()
        v6_loss_meter = AverageMeter()
        v6_correct_meter = AverageMeter()
    start = time.time()

    for step, batch_data in enumerate(test_loader):

        if human_tune:
            data, targets, c10h_val_c10_targets = batch_data
            c10h_val_c10_targets = onehot(c10h_val_c10_targets, 10)
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
                c10h_val_c10_targets = c10h_val_c10_targets.cuda()

        with torch.no_grad():
            outputs = model(data)

        n_obs = data.size(0)

        # compute loss for each test set
        loss = criterion(outputs, targets)
        if human_tune:
            c10h_val_c10_loss = criterion(outputs, c10h_val_c10_targets)

        # save losses
        loss_meter.update(loss.item(), n_obs)
        if human_tune:
            c10h_val_c10_loss_meter.update(c10h_val_c10_loss.item(), n_obs)

        # turn the NN probs into classifications ("predictions")
        _, preds = torch.max(outputs, dim=1)

        if human_tune: 
            _, targets = targets.max(dim=1)
            _, c10h_val_c10_targets = c10h_val_c10_targets.max(dim=1)

        correct_ = preds.eq(targets).sum().item()
        correct_meter.update(correct_, 1)

        if human_tune:
            c10h_val_c10_correct_ = preds.eq(c10h_val_c10_targets).sum().item()
            c10h_val_c10_correct_meter.update(c10h_val_c10_correct_, 1)

    if human_tune:
        # get the full training loss after the last epoch
        # as opposed to the rolling average which is biased
        # due to having many gradient updates in between batches
        # -- get it for c10 labels too (c10h_train_c10_targets)
        for step, (data, targets, c10h_train_c10_targets) in enumerate(train_loader):
            c10h_train_c10_targets = onehot(c10h_train_c10_targets, 10)
            if run_config['use_gpu']:
                data = data.cuda()
                targets = targets.cuda()
                c10h_train_c10_targets = c10h_train_c10_targets.cuda()
            with torch.no_grad(): outputs = model(data)
            n_obs = data.size(0)
            train_loss = criterion(outputs, targets)
            c10h_train_c10_loss = criterion(outputs, c10h_train_c10_targets)
            train_loss_meter.update(train_loss.item(), n_obs)
            c10h_train_c10_loss_meter.update(c10h_train_c10_loss.item(), n_obs)
            _, preds = torch.max(outputs, dim=1)
            _, targets = targets.max(dim=1)
            _, c10h_train_c10_targets = c10h_train_c10_targets.max(dim=1)
            train_correct_ = preds.eq(targets).sum().item()
            c10h_train_c10_correct_ = preds.eq(c10h_train_c10_targets).sum().item()
            train_correct_meter.update(train_correct_, 1)
            c10h_train_c10_correct_meter.update(c10h_train_c10_correct_, 1)

        # TEST 50k cifar10 training to make sure it sustains
        for step, (data, targets) in enumerate(_50k_loader):
            targets = onehot(targets, 10)
            if run_config['use_gpu']:
                data = data.cuda()
                targets = targets.cuda()
            with torch.no_grad(): outputs = model(data)
            n_obs = data.size(0)
            _50k_loss = criterion(outputs, targets)
            _50k_loss_meter.update(_50k_loss.item(), n_obs)
            _, preds = torch.max(outputs, dim=1)
            _, targets = targets.max(dim=1)
            _50k_correct_ = preds.eq(targets).sum().item()
            _50k_correct_meter.update(_50k_correct_, 1)

        # TEST v4/v6 CIFAR 10.1
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
        c10h_val_c10_accuracy = \
            c10h_val_c10_correct_meter.sum / float(len(test_loader.dataset))
        c10h_train_c10_accuracy = \
            c10h_train_c10_correct_meter.sum / float(len(train_loader.dataset))
        train_accuracy = train_correct_meter.sum / float(len(train_loader.dataset))
        _50k_accuracy = _50k_correct_meter.sum / float(len(_50k_loader.dataset))
        v4_accuracy = v4_correct_meter.sum / float(len(v4_loader.dataset))
        v6_accuracy = v6_correct_meter.sum / float(len(v6_loader.dataset))

    if human_tune:
        logger.info('- epoch {}    c10h_train    : {:.4f} (acc: {:.4f}) | c10h_val    : {:.4f} (acc: {:.4f})'.format(
            epoch, train_loss_meter.avg, train_accuracy, loss_meter.avg, accuracy))     
        logger.info('-            c10h_train_c10: {:.4f} (acc: {:.4f}) | c10h_val_c10: {:.4f} (acc: {:.4f})'.format(
            c10h_train_c10_loss_meter.avg, c10h_train_c10_accuracy, c10h_val_c10_loss_meter.avg, c10h_val_c10_accuracy))  
        logger.info('-            v4            : {:.4f} (acc: {:.4f}) |           v6: {:.4f} (acc: {:.4f})'.format(
            v4_loss_meter.avg, v4_accuracy, v6_loss_meter.avg, v6_accuracy))
        logger.info('-            c10_50k       : {:.4f} (acc: {:.4f})'.format(
            _50k_loss_meter.avg, _50k_accuracy))
    else:
        logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if human_tune:
        return {'epoch'          : epoch,

                'c10h_train_loss': train_loss_meter.avg,
                'c10h_train_acc' : train_accuracy,

                'c10h_val_loss': loss_meter.avg,
                'c10h_val_acc' : accuracy,

                'c10h_train_c10_loss': c10h_train_c10_loss_meter.avg,
                'c10h_train_c10_acc' : c10h_train_c10_accuracy,

                'c10h_val_c10_loss': c10h_val_c10_loss_meter.avg,
                'c10h_val_c10_acc' : c10h_val_c10_accuracy,

                'v4_loss': v4_loss_meter.avg,
                'v4_acc' : v4_accuracy,

                'v6_loss': v6_loss_meter.avg,
                'v6_acc' : v6_accuracy,

                'c10_50k_loss': _50k_loss_meter.avg,
                'c10_50k_acc' : _50k_accuracy,                 
        }
    else:
        return accuracy


def main():
    # parse command line argument and generate config dictionary
    config = parse_args()

    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']

    human_tune = run_config['human_tune']
    if human_tune: human_tune_scores = []

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
        train_loader, test_loader, _50k_loader, v4_loader, v6_loader = \
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
        test_criterion = CrossEntropyLoss(size_average=True)
    else:
        test_criterion = nn.CrossEntropyLoss(size_average=True)

    epochs = optim_config['epochs']
    print('expected number (max) epochs:', str(epochs))

    for ep in np.arange(1, epochs + 1): # is there a 0 epoch?    
        temp_eval_file = run_config['resume'].split('best')[0] + 'state_' + str(ep) + '.pth'
        print('temp eval file: ', temp_eval_file)

        # load model
        # load pretrained weights if given
        if os.path.isfile(temp_eval_file):
            print("=> loading checkpoint '{}'".format(temp_eval_file))
            checkpoint = torch.load(temp_eval_file)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(temp_eval_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(temp_eval_file))


        #evaluate model

        if human_tune:
            scores = test(epoch, model, test_criterion,
                    (train_loader, test_loader, _50k_loader, v4_loader, v6_loader),
                    run_config, writer,
                    human_tune=human_tune)
            # print(scores)
            human_tune_scores.append(scores)
        else:
            accuracy = test(epoch, model, test_criterion, test_loader,
                                run_config, writer, human_tune=human_tune)

    # after iterating through epochs and saving scores to list, dump
    # into csv file

    # create output directory
    c10h_outdir = run_config['c10h_scores_outdir']
    if not os.path.exists(c10h_outdir):
        os.makedirs(c10h_outdir)
    
    # resave (overwrite) scores file with latest entries
    keys = human_tune_scores[0].keys()
    print('keys: ', keys)

#    with open(os.path.join(c10h_outdir, 'test_scores.csv'), 'wb') as output_file:
        with open(os.path.join(c10h_outdir, 'test_scores.csv'), 'w') as output_file:    # changed from above
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(human_tune_scores)

if __name__ == '__main__':
    main()
