#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=battleday@princeton.edu
#SBATCH --mem=16000
# activate conda environment
source activate pytorch_env

#SBATCH --output=test_output.out

echo 'entering python script'

python ./main.py --arch vgg --seed 7 --outdir /scratch/ruairidh/results/vgg_15_BN_64/

echo 'done'
