use_gpu = True

# EXTERNAL LIBRARY IMPORTS
import numpy as np 
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import torch # Need torch version >=0.3
import torch.nn as nn 
import torch.optim as optim 
assert float(torch.__version__[:3]) >= 0.3

# MISTER ED SPECIFIC IMPORT BLOCK
# (here we do things so relative imports work )
# Universal import block 
# Block to get the relative imports working 
import os, sys, re, gc, pickle 
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


import config
import prebuilt_loss_functions as plf
import loss_functions as lf 
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import adversarial_training as advtrain
import adversarial_evaluation as adveval
import utils.checkpoints as checkpoints
import adversarial_perturbations as ap 
import adversarial_attacks as aa
import spatial_transformers as st

### START pytorch_image_classification imports
import time, random, json, logging, argparse, csv
from pytorch_image_classification_dataloader_c10h import get_loader
from pytorch_image_classification_utils import (str2bool, load_model, save_checkpoint, create_optimizer,
                                                AverageMeter, mixup, CrossEntropyLoss, onehot)
# from rutils_run import save_checkpoint_epoch
from pytorch_image_classification_argparser import get_config

sys.argv = ['']

def parse_args(arch, mdl_config):

    parser = argparse.ArgumentParser()
    # parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--arch', type=str, default=arch)
    # parser.add_argument('--config', type=str, default='tmp_reference_model/resnet_basic_110_config.json')
    parser.add_argument('--config', type=str, default=mdl_config)
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
    parser.add_argument('--batch_size', type=int, default=500)
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

_, normalizer = cifar_loader.load_pretrained_cifar_resnet(flavor=20,
                                                          return_normalizer=True)
del _

root = '/tigress/joshuacp/model_results/optimal_basic_tuning_josh_alt_controls/'

def get_arch(name):
    if 'vgg_15_BN_64' in name: return 'vgg'
    if 'resnet_basic_110' in name: return 'resnet' 
    if 'resnet_preact_bottleneck_164' in name: return 'resnet_preact' 
    if 'wrn_28_10' in name: return 'wrn' 
    if 'densenet_BC_100_12' in name: return 'densenet' 
    if 'pyramidnet_basic_110_270' in name: return 'pyramidnet' 
    if 'resnext_29_8x64d' in name: return 'resnext' 
    if 'shake_shake_26_2x64d_SSI_cutout16' in name: return 'shake_shake'
    return 'NOT FOUND!!!'

def get_meta(verbose=False, filt=None):
    model_meta = []
    for folder in os.listdir(root):
        if filt:
            if filt not in folder:
                continue
        param_folders = os.listdir(os.path.join(root, folder))
        for param_folder in param_folders:
            pth_file = os.listdir(os.path.join(root, folder, param_folder))[0]
            meta = {}
            meta['arch'] = get_arch(folder)
            meta['model_name'] = folder
            meta['param_folder'] = param_folder
            meta['pth_file'] = pth_file
            if 'con_True' in param_folder:
                meta['human'] = False
            else:
                meta['human'] = True  

            model_meta.append(meta)
            if verbose: print(meta)
    return model_meta

# model_meta = get_meta(verbose=True, filt='shake_shake')
# model_meta = get_meta(verbose=False)


# model_meta = get_meta(verbose=False, filt='dense')
model_meta = get_meta(verbose=False)

config_folder = '/tigress/ruairidh/model_results/optimal_training_run'
break_loop = False
epochs_per_model = 10 #10
verbosity='low'# 'snoop' # 'low'

batch_size = 128
n_val_batches = int(np.ceil(10000/float(batch_size)))
#n_val_batches = 100
save_results = True

for mm in model_meta:
    
    print(mm['model_name'], mm['human'])
    print('')
    
    config = parse_args(arch=mm['arch'], 
                        mdl_config=os.path.join(config_folder,
                                                mm['model_name'],
                                                'config.json'))
    
    resume_path = os.path.join(root, 
                               mm['model_name'],
                               mm['param_folder'],
                               mm['pth_file'])

    model = load_our_model(config, resume_path)
    
    delta_threat = ap.ThreatModel(ap.DeltaAddition, 
                              {'lp_style': 'inf', 
                               'lp_bound': 8.0 / 255})
    attack_loss = plf.VanillaXentropy(model, normalizer)
    attack_object = aa.FGSM(model, normalizer, delta_threat, attack_loss)
    attack_kwargs = {'verbose': False} # kwargs to be called in attack_object.attack(...)
    attack_params = advtrain.AdversarialAttackParameters(attack_object, proportion_attacked=0.2, 
                                                         attack_specific_params={'attack_kwargs': attack_kwargs})
    experiment_name = 'test'
    architecture = 'test'
    training_obj = advtrain.AdversarialTraining(model, normalizer, experiment_name, architecture)
    
    train_loss = nn.CrossEntropyLoss() # just use standard XEntropy to train
    
    for epoch in range(epochs_per_model + 1):
        
        cifar_trainset = cifar_loader.load_cifar_data('val', batch_size=batch_size)
        print('WARNING: using validation as train and val currently as a test!! turn off later!')
        cifar_valset = cifar_loader.load_cifar_data('val', batch_size=batch_size)
        
        if epoch != 0:
            training_obj.train(cifar_trainset, 1, train_loss, 
                               attack_parameters=attack_params,
                               verbosity=verbosity)

        ### EVAL !!! ###
        adv_eval_object = adveval.AdversarialEvaluation(model, normalizer)
        to_eval_dict = {'top1': 'top1', 
                        'avg_loss_value': 'avg_loss_value'}

        
        #------ FGSM8 Block 
        linf_8_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf', 
                                                         'lp_bound': 8.0 / 255.0})
        fgsm8_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf', 
                                                         'lp_bound': 8.0/ 255.0})
        fgsm8_attack_loss = plf.VanillaXentropy(model, normalizer)
        fgsm8_attack = aa.FGSM(model, normalizer, linf_8_threat, fgsm8_attack_loss)
        fgsm8_attack_kwargs = {'verbose': False}
        fgsm8_attack_params = advtrain.AdversarialAttackParameters(fgsm8_attack,
                                                                   attack_specific_params=
                                                                   {'attack_kwargs': fgsm8_attack_kwargs})
        
        fgsm8_eval = adveval.EvaluationResult(fgsm8_attack_params, 
                                              to_eval=to_eval_dict)
        attack_ensemble = {'fgsm8': fgsm8_eval}

        ensemble_out = adv_eval_object.evaluate_ensemble(cifar_valset, attack_ensemble, 
                                                         verbose=False, 
                                                         num_minibatches=n_val_batches)



        sort_order = {'ground': 1, 'fgsm8': 2, 'pgd4': 3, 'pgd8': 4}
        def pretty_printer(eval_ensemble, result_type):
            print('~' * 10, result_type, '~' * 10)
            for key in sorted(list(eval_ensemble.keys()), key=lambda k: sort_order[k]):
                eval_result = eval_ensemble[key]
                pad = 6 - len(key)
                if result_type not in eval_result.results:
                    continue 
                avg_result = eval_result.results[result_type].avg
                print(key, pad* ' ', ': ', avg_result)

        print('')
        pretty_printer(ensemble_out, 'top1')
        pretty_printer(ensemble_out, 'avg_loss_value')
        print('')
        print('')

        for ae_key in attack_ensemble.keys():
            if ae_key not in mm.keys():
                mm[ae_key] = {}
            for e_key in to_eval_dict.keys():
                if e_key not in mm[ae_key].keys() and (e_key in ensemble_out[ae_key].results.keys()):
                    mm[ae_key][e_key] = []
                try:
                    mm[ae_key][e_key].append(ensemble_out[ae_key].results[e_key].avg)
                except:
                    pass
        print(mm)

        if break_loop: break
        if save_results: 
            pickle.dump(model_meta, open('fgsm_defense_results.pickle','wb'))
            
    del model
    del ensemble_out, training_obj, fgsm8_eval
    del cifar_trainset, cifar_valset
    gc.collect()




