#!/usr/bin/bash
#SBATCH --job-name=WS_h
#SBATCH --error=./logs/WS1_h_%a.err
#SBATCH --output=./logs/WS1_h_%a.out
#SBATCH --array=0-rep
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c 64
#SBATCH --mem=8GB

ml python/3.12.1
time python3 ./sendOut.py ${SLURM_ARRAY_TASK_ID} 1