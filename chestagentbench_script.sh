#!/bin/bash

#SBATCH --job-name=chestagentbench
#SBATCH -c 4
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=16:00:00
#SBATCH --mem=50G
#SBATCH --output=chestagentbench-%j.out
#SBATCH --error=chestagentbench-%j.err

module load arrow clang/18.1.8 scipy-stack

source venv/bin/activate

/scratch/lijunzh3/MedRAX2/venv/bin/python -m benchmarking.cli run --benchmark chestagentbench --provider google --model gemini-2.5-pro --system-prompt CHESTAGENTBENCH_PROMPT --data-dir benchmarking/data/chestagentbench --output-dir temp --max-questions 20 --temperature 0.7 --top-p 0.95 --max-tokens 5000 --concurrency 4 --random-seed 42