#!/usr/bin/env bash

# activate conda environment
echo 'outer: activating env'
source activate pytorch_env
model=wrn_28_10
epoch=34
token=wrn
identifier=con_True_lr_0.001_seed_0
echo ${token}
echo ${model}
echo 'running sbatch'
sbatch --output=${model}.out --export=token=${token},model=${model},identifier=${identifier} run_extract_softmax_inner_post.sh
echo 'outer done'
