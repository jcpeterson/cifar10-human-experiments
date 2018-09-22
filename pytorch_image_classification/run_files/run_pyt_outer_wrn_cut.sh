#!/usr/bin/env bash
echo 'outer script'

time=4310
model=wrn_28_10_cutout16
log_file=${model}.out
out_dir=${model}
python_args="--arch=wrn --depth 28 --epochs=200 --scheduler=cosine --base_lr=0.1 --batch_size=64"
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
