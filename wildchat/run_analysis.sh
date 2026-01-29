#!/bin/bash
#SBATCH --job-name="collect_wildchat_queries"
#SBATCH --time=5:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=collecting-queries-v2.out


python collect_queries.py