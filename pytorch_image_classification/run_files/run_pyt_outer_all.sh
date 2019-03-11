#!/usr/bin/env bash
echo 'outer script'

models='vgg_15_BN_64'
#resnet_basic_110
#resnet_preact_bottleneck_164
#densenet_BC_100_12
#wrn_28_10
#pyramidnet_basic_110_270
#resnext_29_8x64d
#wrn_28_10_cutout16
#shake_shake_26_2x64d_SSI_cutout16' # no commas!
# maybe no quotes needed below?

declare -A params=([vgg_15_BN_64]="--arch=vgg" 
[resnet_basic_110]="--arch=resnet --depth 110 --block_type basic" 
[resnet_preact_bottleneck_164]="--arch=resnet_preact --depth 164 --block_type bottleneck" 
[wrn_28_10]="--arch=wrn --depth 28 --widening_factor 10" 
[densenet_BC_100_12]="--arch=densenet --depth 100 --block_type=bottleneck --growth_rate=12 --compression_rate=0.5" 
[pyramidnet_basic_110_270]="--arch=pyramidnet --depth 110 --block_type=basic --pyramid_alpha=270" 
[resnext_29_8x64d]="--arch=resnext --depth 29 --cardinality=8 --base_channels=64 --batch_size=64 --base_lr=0.05"
[wrn_28_10_cutout16]="--arch=wrn --depth 28 --epochs=200 --scheduler=cosine --base_lr=0.1 --batch_size=64 --seed 17 --use_cutout --cutout_size 16" 
[shake_shake_26_2x64d_SSI_cutout16]="--arch=shake_shake --depth=26 --base_channels=64 --epochs=300 --scheduler=cosine --base_lr=0.1 --seed 17 --batch_size=64 --use_cutout --cutout_size=16")

declare -A times=([vgg_15_BN_64]=360 
[resnet_basic_110]=360 
[resnet_preact_bottleneck_164]=1440 
[wrn_28_10]=1440 
[densenet_BC_100_12]=1440 
[pyramidnet_basic_110_270]=4320 
[resnext_29_8x64d]=4320 
[wrn_28_10_cutout16]=4320 
[shake_shake_26_2x64d_SSI_cutout16]=4320)
echo 'declared dicts'

option=''
held_outs='6'

for model in $models
  do
  echo $model
  # might need extra curlies
  time=${times[$model]}
  python_args=${params[$model]}
  echo $time
  echo $python_args
  for held_out in $held_outs
  do
    echo 'held out'
    echo $held_out
    log_file="${model}${option}_${held_out}".out
    out_dir="${model}${option}_${held_out}"
    sbatch --time=${time} --job-name=${model} --output=${log_file} --export=held_out=${held_out},out_dir=${out_dir},python_args="${python_args}",log_file=${log_file},option=$option run_pyt_inner.sh
  done
  echo 'done outer'
  done
