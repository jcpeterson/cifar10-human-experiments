#!/usr/bin/env bash
echo 'outer script'

time=4310
model=resnext_29_8x64d
log_file=${model}.out
out_dir=${model}
python_args="--arch=resnext --depth 29 --cardinality=8 --base_channels=64 --batch_size=64 --base_lr=0.05"
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
