#!/usr/bin/env bash
# serial job using 1 core for 4 hours (max)
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=battleday@princeton.edu

# activate conda environment
source activate pytorch_env
SDIR='/scratch/ruairidh'
mkdir ${SDIR}
mkdir ${SDIR}/results

python ./main.py 

echo 'done'
