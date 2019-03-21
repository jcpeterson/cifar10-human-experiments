#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000


echo 'entering inner script'
echo 'activating virtual env'
# activate virtual environment here
# source activate pytorch_env
echo $PWD
cd ..
echo $PWD
SDIR='/tigress/joshuacp/model_results'
O_DIR='9k_tuning_run_1_smoothed_temp2'
echo ${model}
echo 'identifier: '${identifier}
echo ${python_args}
echo ${logfile}
echo 'entering python script'
SV_DIR="${SDIR}/${O_DIR}/${model}/${identifier}"
echo ${SV_DIR}

#python -u ./tune_with_cifar10h.py ${python_args} --resume=${resume} --c10h_scores_outdir=${SV_DIR} --config=${config}
python -u ./train_with_cifar10h_smoothed.py ${python_args} --c10h_scores_outdir=${SV_DIR}
cd .
echo $PWD
cp ./${logfile} ${SDIR}/${O_DIR}/${model}/.

echo 'inner done'

