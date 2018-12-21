#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=17000
#SBATCH --time=360

python cvpr_fast_fgsm_defense_training.py 

