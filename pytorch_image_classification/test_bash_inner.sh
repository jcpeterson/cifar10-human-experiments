#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=10


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
SV_DIR="${SDIR}/run_1/test_saves/${model}"
echo ${resume}
echo ${SV_DIR}


python -u ./test_with_cifar10h.py ${python_args} --resume=${resume} --c10h_scores_outdir=${SV_DIR}

cp ./${logfile} ${SDIR}/run_1/test_saves/${model}.

echo 'inner done'

