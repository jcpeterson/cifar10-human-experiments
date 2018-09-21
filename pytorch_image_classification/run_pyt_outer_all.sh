#!/usr/bin/env bash
echo 'outer script'



declare -A times=( [4_1]=4_2 [5_1]=5_2 [6_1]=6_2 [7_1]=7_2 [8_1]=8_2 )

for i in "${!pairs[@]}"; do
  j=${pairs[$i]}

time=60
model=vgg_15_BN_64
log_file=${model}.out
out_dir=${model}
python_args='--arch=vgg'
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args=${python_args},log_file=$log_file run_pyt_inner.sh
