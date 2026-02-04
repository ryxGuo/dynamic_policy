#!/bin/bash
#SBATCH --job-name="collect_wildchat_queries"
#SBATCH --time=5:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=collecting-queries-v2.out

CONFIG_FILE="${1:-config.csv}"

# Read CSV file and run for each row
while IFS=',' read -r version prompt_style model_name index_start index_end batch_size max_length output_dir; do
    # Skip header line
    [[ "$version" == "version" ]] && continue
    
    echo "Running with: $version | $prompt_style | $model_name | $index_start-$index_end"
    
    python query_filter.py \
        --version "$version" \
        --prompt_style "$prompt_style" \
        --model_name "$model_name" \
        --index_start "$index_start" \
        --index_end "$index_end" \
        --batch_size "$batch_size" \
        --max_length "$max_length" \
        --output_dir "$output_dir"
done < "$CONFIG_FILE"