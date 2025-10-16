#!/bin/bash

#SBATCH --job-name=medgemma
#SBATCH -c 4
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=16:00:00
#SBATCH --mem=50G
#SBATCH --output=medgemma-%j.out
#SBATCH --error=medgemma-%j.err

cd medrax/tools/vqa/medgemma

source medgemma/bin/activate

python medgemma.py