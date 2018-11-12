#!/usr/bin/env bash
echo 'outer script'

time=8630
model=shake_shake_26_2x64d_SSI_cutout16
log_file=${model}.out
out_dir=${model}
python_args="--arch=shake_shake --depth=26 --base_channels=64 --epochs=300 --scheduler=cosine --base_lr=0.1 --seed 17 --batch_size=64 --use_cutout --cutout_size=16"
sbatch --time=${time} --output=${log_file} --export=out_dir=${out_dir},python_args="${python_args}",log_file=$log_file run_pyt_inner.sh
