#!/bin/bash
#SBATCH --job-name="collect_wildchat_queries"
#SBATCH --time=5:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=logs/collecting-queries-10k-maxlen300.out


version="version-final-maxlen-300"
#prompt_style="style1_revised_explicit_safety"
prompt_style="style1_revised"
model_name="Qwen/Qwen2.5-7B-Instruct"
index_start=0
index_end=10000
batch_size=8
max_length=300
output_dir="./wildchat-query-results"

python query_filter.py \
        --version "$version" \
        --prompt_style "$prompt_style" \
        --model_name "$model_name" \
        --index_start "$index_start" \
        --index_end "$index_end" \
        --batch_size "$batch_size" \
        --max_length "$max_length" \
        --output_dir "$output_dir"