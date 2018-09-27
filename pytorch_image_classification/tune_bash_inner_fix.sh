#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=1440


echo 'entering inner script'
echo 'activating virtual env'
# activate virtual environment here
# source activate pytorch_env

SDIR='/tigress/ruairidh/model_results'
JDIR='/tigress/joshuacp/model_results'
echo ${model}
echo 'identifier: '${identifier}
echo ${python_args}
echo ${logfile}
echo 'entering python script'
resume="${SDIR}/run_2/${model}/model_best_state.pth"
SV_DIR="${JDIR}/run_2/saves/${model}/${identifier}"
config="${SDIR}/run_2/${model}/config.json"
echo ${resume}
echo ${SV_DIR}


python -u ./tune_with_cifar10h.py ${python_args} --resume=${resume} --c10h_scores_outdir=${SV_DIR} --config=${config}

cp ./${logfile} ${JDIR}/run_2/saves/${model}/.

echo 'inner done'

