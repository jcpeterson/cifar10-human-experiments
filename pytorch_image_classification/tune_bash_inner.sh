#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=59


echo 'entering inner script'
echo 'activating virtual env'
# activate virtual environment here
source activate pytorch_env

SDIR='/tigress/ruairidh/model_results'
echo ${model}
echo ${python_args}
echo ${logfile}
echo 'entering python script'
resume="${SDIR}/run_1/${model}/model_best_state.pth"
echo ${resume}
python -u ./tune_with_cifar10h.py ${python_args} --resume=${resume} # do we need outdir? --outdir ${SDIR}/${out_d$

cp ./${log_file} ${SDIR}/run_1/.

echo 'inner done'

