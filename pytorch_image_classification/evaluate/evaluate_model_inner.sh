#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=350


echo 'entering inner script'
echo 'activating virtual env'
# activate virtual environment here
source activate pytorch_env

SDIR='/tigress/ruairidh/model_results'
echo ${model}
echo 'run: '${run}
echo ${python_args}
echo ${logfile}
echo 'entering python script'
resume="${SDIR}/full_training_runs/training_run_${run}/${model}/model_best_state.pth"
config="${SDIR}/full_training_runs/training_run_${run}/${model}/config.json"
echo ${resume}

python -u ./evaluate_model_all_datasets.py ${python_args} --resume=${resume} --c10h_scores_outdir='nonsense' --config=${config}
echo 'training_run: '${run}
echo 'inner done'

