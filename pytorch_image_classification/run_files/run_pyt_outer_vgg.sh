#!/usr/bin/env bash
echo 'outer script'

time=350
model=vgg_15_BN_64
log_file=${model}.out
out_dir=${model}
python_args='--arch=vgg\ --resume=False'
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
