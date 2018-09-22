#!/usr/bin/env bash
echo 'outer script'

time=350
model=resnet_preact_bottleneck_164
log_file=${model}.out
out_dir=${model}
python_args="--arch=resnet_preact --depth 164 --block_type bottleneck"
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
