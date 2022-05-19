#!/bin/bash
#SBATCH --job-name=ts10c4
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/hipo/exp10/c4-out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/hipo/exp10/c4-err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/11.1.1-GCC-10.2.0

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate base
python /hits/basement/nlp/fatimamh/codes/HipoRank-master/exp10_ts_c4.py 

