#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=20

# activate conda environment
echo 'inner: activating env'
source activate pytorch_env
echo ${model}
echo ${con}
echo ${python_args}
d_path="/tigress/ruairidh/model_results"
s_path="/tigress/ruairidh/model_results/preds_scores_9k"
resume_path="${d_path}/optimal_9k/${con}/${model}/fold_${fold}/model_best_state_c10h_val_c10_acc.pth"
config_path="${d_path}/optimal_training_run/${model}/config.json"
cd ..
echo $PWD
python predict_and_score_with_cifar10h_2.py ${python_args} --config ${config_path} --resume ${resume_path} --outdir /home/ruairidh/superman/cifar10-human-experiments/predictions/post --gpu 0
cd .
echo $PWD
echo 'inner done'
