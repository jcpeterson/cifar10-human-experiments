#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:1
python cifar10_human_tuning.py