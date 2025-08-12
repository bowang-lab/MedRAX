#!/bin/bash

#SBATCH --job-name=chestagentbench
#SBATCH -c 4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --exclude=gpu138
#SBATCH --time=16:00:00
#SBATCH --mem=50G
#SBATCH --output=chestagentbench-%j.out
#SBATCH --error=chestagentbench-%j.err

source venv/bin/activate

python -m benchmarking.cli run --model gpt-5 --provider medrax --system-prompt CHESTAGENTBENCH_PROMPT --benchmark chestagentbench --data-dir /scratch/ssd004/scratch/victorli/chestagentbench --output-dir temp --max-questions 2500 --concurrency 4