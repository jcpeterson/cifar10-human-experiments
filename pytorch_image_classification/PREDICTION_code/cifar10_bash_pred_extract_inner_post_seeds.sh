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
echo 'seed'
echo ${seed}
echo ${python_args}

echo ${con}
d_path="/tigress/ruairidh/model_results/optimal_basic_tuning_all_seeds/${con}/seed_${seed}/${model}"
s_path="/tigress/ruairidh/model_results/preds_scores_50k_seeds/${con}"
resume_path="${d_path}/model_best_state_c10h_val_c10_acc.pth"
config_path="/tigress/ruairidh/model_results/optimal_training_run/${model}/config.json"
cd ..
echo $PWD
python predict_and_score_with_cifar10_seeds.py ${python_args} --config ${config_path} --resume ${resume_path} --outdir ${s_path} --c10h_scores_outdir ${s_path} --gpu 0
cd .
echo $PWD
echo 'inner done'
