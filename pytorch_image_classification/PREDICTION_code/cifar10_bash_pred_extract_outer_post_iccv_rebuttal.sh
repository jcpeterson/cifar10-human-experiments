#!/usr/bin/env bash

# activate conda environment
models='shake_shake_26_2x64d_SSI_cutout16' # no commas!
# maybe no quotes needed below?

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

  for fold in "${folds[@]}"
  do

      echo ${model}
      echo ${arch}
      echo ${fold}
      echo 'running sbatch'
      python_args="--arch=${arch} --dataset=CIFAR10H --no_output --c10h_datasplit_seed=0 --human_tune --cv_index=${fold}"
      echo 'python args: '"${python_args}"

      sbatch --output=pretrained.pred_extract.${model}.out --export=python_args="${python_args}",model=${model} cifar10_bash_pred_extract_inner_post.sh


    done

  done
echo 'outer done'
