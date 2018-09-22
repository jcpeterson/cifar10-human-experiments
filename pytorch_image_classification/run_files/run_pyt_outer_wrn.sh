#!/usr/bin/env bash
echo 'outer script'

time=1430
model=wrn_28_10
log_file=${model}.out
out_dir=${model}
python_args="--arch=wrn --depth 28 --widening_factor 10"
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
