#!/usr/bin/env bash

# activate conda environment
#models='vgg_15_BN_64'
models='resnet_basic_110'
#resnet_preact_bottleneck_164
#densenet_BC_100_12
#wrn_28_10
#pyramidnet_basic_110_270
#resnext_29_8x64d
#wrn_28_10_cutout16
#shake_shake_26_2x64d_SSI_cutout16' # no commas!
# maybe no quotes needed below?

#declare -a control=(True False)
declare -a control=(True False)

#declare -a folds=(0 1 2 3 4 5 6 7 8 9)
declare -a folds=(0)

declare -A archs=([vgg_15_BN_64]=vgg [resnet_basic_110]=resnet 
[resnet_preact_bottleneck_164]=resnet_preact 
[wrn_28_10]=wrn [densenet_BC_100_12]=densenet 
[pyramidnet_basic_110_270]=pyramidnet [resnext_29_8x64d]=resnext 
[wrn_28_10_cutout16]=wrn [shake_shake_26_2x64d_SSI_cutout16]=shake_shake)

for model in $models
  do

  arch=${archs[$model]} 

  for con in "${control[@]}"
  do

    for fold in "${folds[@]}"
    do

      echo ${model}
      echo ${arch}
      echo ${con}
      echo ${fold}
      echo 'running sbatch'
      python_args="--arch=${arch} --dataset=CIFAR10H --no_output --c10h_datasplit_seed=0 --human_tune --nonhuman_control=${con} --cv_index=${fold}"
      echo 'python args: '"${python_args}"

      sbatch --output=pred_extract.${model}.con.${con}.post.out --export=python_args="${python_args}",model=${model},con=${con} bash_pred_extract_inner.sh

      done

    done

  done
echo 'outer done'
