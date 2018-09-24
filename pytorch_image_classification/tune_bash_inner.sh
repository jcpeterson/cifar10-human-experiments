#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=59

# activate virtual environment here
source activate pytorch_env

SDIR='/tigress/ruairidh/model_results'
echo ${model}
echo ${python_args}
echo ${log_file}
echo 'entering python script'
resume="${SDIR}/run_1/${model}/model_best_state.pth"
python -u ./tune_with_cifarh10.py ${python_args} ${resume} # do we need outdir? --outdir ${SDIR}/${out_d$

cp ./${log_file} ${SDIR}/run_1/.

echo 'done'

