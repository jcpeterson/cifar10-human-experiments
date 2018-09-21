#!/usr/bin/env bash
echo 'outer script'


declare -A models=( [vgg]=vgg_15_BN_64 [res_b]=resnet_basic_110)
declare -A times=( [vgg]=350 [res_b]=350 )
declare -A params=([vgg]='--arch vgg --seed/ 7' [res_b]='--arch resnet --depth 110 --block_type basic --seed 7')
echo 'declared dicts'

for token in "${!models[@]}"
  do
  echo ${token}
  model=${models[${token}]}   # need curly braces?
  time=${times[${token}]}
  python_args=${params[${token}]}

  log_file=${model}.out
  out_dir=${model}
  sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args=${python_args},log_file=${log_file} run_pyt_inner.sh
  echo 'done outer'
  done
