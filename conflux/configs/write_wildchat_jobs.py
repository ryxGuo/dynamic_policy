"""
Auto-generated config writer for job_configs.csv with fixed values.
You can fill `config` manually or load from a JSON/YAML file.
"""
import argparse
import csv
import os
import json
try:
    import yaml
except ImportError:
    yaml = None  # YAML support is optional


# === Option 1: Fill manually ===
config = {
    'version': [],
    'prompt_style': [],
    'model_name': [],
    'index_start': [],
    'index_end': [],
    'max_length': [],
}
# === Fixed values ===
fixed_values = {
    'batch_size': '8',
    'output_dir': './wildchat_query_results',
}


# === Option 2: Load from JSON or YAML ===
def load_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    ext = os.path.splitext(file_path)[-1].lower()
    with open(file_path, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError("PyYAML is not installed. Run `pip install pyyaml`.")
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

def validate_config(config):
    lengths = [len(v) for v in config.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent config list lengths: {lengths}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, help="Path to where to write the CSV file", default="job_configs.csv")
    parser.add_argument("--config_file", type=str, help="Path to a JSON/YAML config file")
    args = parser.parse_args()

    if args.config_file:
        config = load_from_file(args.config_file)
    
    validate_config(config)

    # === Write to CSV ===
    with open(args.output_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['version', 'prompt_style', 'model_name', 'index_start', 'index_end', 'max_length', 'batch_size', 'output_dir'])
        writer.writeheader()
        for i in range(len(list(config.values())[0])):
            row = {k: v[i] for k, v in config.items()}
            row.update(fixed_values)  # Add fixed values
            writer.writerow(row)

    print(f"✅ {args.output_file_path}.csv written successfully.")
