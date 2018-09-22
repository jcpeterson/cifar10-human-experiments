#!/usr/bin/env bash
echo 'outer script'

time=1430
model=densenet_BC_100_12
log_file=${model}.out
out_dir=${model}
python_args="--arch=densenet --depth 100 --block_type=bottleneck --growth_rate=12 --compression_rate=0.5"
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
