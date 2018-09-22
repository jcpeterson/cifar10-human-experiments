#!/usr/bin/env bash
echo 'outer script'

time=4310
model=pyramidnet_basic_110_270
log_file=${model}.out
out_dir=${model}
python_args="--arch=pyramidnet --depth 110 --block_type=basic --pyramid_alpha=270"
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
