#!/usr/bin/env bash

# activate conda environment
echo 'outer: activating env'
source activate pytorch_env
declare -A models=([vgg]=vgg_15_BN_64 [resnet]=resnet_basic_110 [resnet_preact]=resnet_preact_bottleneck_164  [wrn]=wrn_28_10 [densenet]=densenet_BC_100_12 [pyramidnet]=/pyramidnet_basic_110_270 [resnext]=resnext_29_8x64d [wrn]=wrn_28_10_cutout16 [shake_shake]=shake_shake_26_2x64d_SSI_cutout16)

  token=
  identifier=
  echo ${token}
  model=${models[${token}]} 
  echo ${model}
  echo 'running sbatch'
  sbatch --output=${model}.out --export=token=${token},model=${model},identifier=${identifier} run_extract_softmax_inner_post.sh
echo 'outer done'
