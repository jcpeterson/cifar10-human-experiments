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

python extract_softmax.py --dataset CIFAR10 --arch ${token} --config /tigress/ruairidh/model_results/optimal_training_run/${model}/config.json --resume  /tigress/ruairidh/model_results/optimal_training_run/${model}/model_best_state.pth --outdir /home/ruairidh/superman/cifar10-human-experiments/predictions --gpu 0

echo 'inner done'
