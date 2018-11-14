#!/usr/bin/env bash

# activate conda environment
models='pyramidnet_basic_110_270 shake_shake_26_2x64d_SSI_cutout16 resnext_29_8x64d densenet_BC_100_12 wrn_28_10 resnet_preact_bottleneck_164 resnet_basic_110 vgg_15_BN_64'
#models='densenet_BC_100_12'

declare -A archs=([vgg_15_BN_64]=vgg [resnet_basic_110]=resnet [resnet_preact_bottleneck_164]=resnet_preact [wrn_28_10]=wrn [densenet_BC_100_12]=densenet [pyramidnet_basic_110_270]=pyramidnet [resnext_29_8x64d]=resnext [wrn_28_10_cutout16]=wrn [shake_shake_26_2x64d_SSI_cutout16]=shake_shake)

for model in $models
  do
  token=${archs[$model]} 
  echo ${token}
  echo ${model}
  echo 'running sbatch'
  sbatch --output=${model}.post.out --export=token=${token},model=${model} run_extract_softmax_inner_post.sh
  done
echo 'outer done'
