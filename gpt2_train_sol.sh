#!/bin/bash
#SBATCH -p general
#SBATCH -t 5-00:0:00
#SBATCH --mem=100GB
#SBATCH -G a100:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sshah205@asu.edu
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE

module purge
module load mamba/latest
source activate /scratch/phegde7/.conda/envs/attention_guidance_v2/

python gpt2_train.py
