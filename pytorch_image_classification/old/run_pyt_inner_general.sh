#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=battleday@princeton.edu
#SBATCH --mem=16000
#SBATCH --output=vgg_15_BN_64.out

# activate conda environment
source activate pytorch_env

SDIR='/tigress/ruairidh/model_results'
model='vgg'
params='_15_BN_64'

echo ${SDIR}


echo 'entering python script'

python -u ./main.py --arch ${model} --seed 7 --outdir ${SDIR}/${model}${params} 

cp ./vgg_15_BN_64.out ${SDIR}/.

echo 'done'
