#!/usr/bin/env bash
echo 'outer tuning script'

models=['vgg_15_BN_64' 'resnet_basic_110' 'resnet_preact_bottleneck_164' 'wrn_28_10' 'densenet_BC_100_12' 'pyramidnet_basic_110_270' 'resnext_29_8x64d' 'wrn_28_10_cutout16' 'shake_shake_26_2x64d_SSI_cutout16']
for model in models
   do

