#!/bin/bash

#SBATCH --job-name=medgemma
#SBATCH -c 4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --exclude=gpu138
#SBATCH --time=16:00:00
#SBATCH --mem=50G
#SBATCH --output=medgemma-%j.out
#SBATCH --error=medgemma-%j.err

source medgemma/bin/activate

cd medrax/tools/vqa/medgemma && python medgemma.py