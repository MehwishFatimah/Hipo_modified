#!/bin/bash
#SBATCH --job-name=ex05cl
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/hipo_1/exp05/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/hipo_1/exp05/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p,pascal-crunch.p,ice-deep.p

module load CUDA/11.1.1-GCC-10.2.0

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate hipo_new
python /hits/basement/nlp/fatimamh/codes/HipoRank-master/exp5_c_run.py 

