#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

# activate conda environment
source activate pytorch_env

SDIR='/tigress/ruairidh/model_results'
echo ${SDIR}
echo ${python_args}
echo ${log_file}
echo 'fold'
echo ${cv_index}
echo 'option'
echo ${option}
echo 'entering python script'

python -u "./main${option}.py" ${python_args} --seed 7 --outdir ${SDIR}/${out_dir} --cv_index ${cv_index} --dataset=CIFAR10H

cp ./${log_file} ${SDIR}/.

echo 'done'
