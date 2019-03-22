#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=70

# activate conda environment
echo 'inner: activating env'
source activate pytorch_env
echo ${model}
echo ${con}
echo ${python_args}
d_path="/tigress/ruairidh/model_results"
s_path="/tigress/ruairidh/model_results/preds_scores_9k"
resume_path="${d_path}/optimal_9k/${con}/${model}"
#/model_best_state_c10h_val_c10_acc.pth
echo 'change config path when proper run done'
config_path="${d_path}/optimal_training_run/${model}/config.json"
cd ..
echo $PWD
python predict_and_score_with_cifar10h.py ${python_args} --config ${config_path} --resume ${resume_path} --outdir ${s_path} --c10h_scores_outdir ${s_path} --gpu 0
cd .
echo $PWD
echo 'inner done'
