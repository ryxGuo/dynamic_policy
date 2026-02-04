
def generate_writer_script(headers, output_file="write_job_config.py"):
    import_lines = [
        "import argparse",
        "import csv",
        "import os",
        "import json",
        "try:",
        "    import yaml",
        "except ImportError:",
        "    yaml = None  # YAML support is optional"
    ]

    config_block = "\n".join([f"    '{h}': []," for h in headers])
    header_list = ", ".join([f"'{h}'" for h in headers])

    script = f'''"""
Auto-generated config writer for job_configs.csv.
You can fill `config` manually or load from a JSON/YAML file.
"""

{chr(10).join(import_lines)}

# === Option 1: Fill manually ===
config = {{
{config_block}
}}

# === Option 2: Load from JSON or YAML ===
def load_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{{file_path}}' not found.")
    
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
        raise ValueError(f"Inconsistent config list lengths: {{lengths}}")

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
        writer = csv.DictWriter(f, fieldnames=[{header_list}])
        writer.writeheader()
        for i in range(len(list(config.values())[0])):
            writer.writerow({{k: v[i] for k, v in config.items()}})

    print(f"✅ {{output_file}}.csv written successfully.")
'''

    with open(output_file, "w") as f:
        f.write(script)

    print(f"📝 Generated '{output_file}' with headers: {headers}")

# Write the same version of the previous function, but ensure there is handling for 'fixed' values whose column values don't change
def generate_writer_script_fixed(headers, fixed_values, output_file="write_job_config.py"):
    import_lines = [
        "import argparse",
        "import csv",
        "import os",
        "import json",
        "try:",
        "    import yaml",
        "except ImportError:",
        "    yaml = None  # YAML support is optional"
    ]

    config_block = "\n".join([f"    '{h}': []," for h in headers])
    header_list = ", ".join([f"'{h}'" for h in headers + list(fixed_values.keys())])

    fixed_block = "\n".join([f"    '{k}': '{v}'," for k, v in fixed_values.items()])

    script = f'''"""
Auto-generated config writer for job_configs.csv with fixed values.
You can fill `config` manually or load from a JSON/YAML file.
"""
{chr(10).join(import_lines)}


# === Option 1: Fill manually ===
config = {{
{config_block}
}}
# === Fixed values ===
fixed_values = {{
{fixed_block}
}}


# === Option 2: Load from JSON or YAML ===
def load_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{{file_path}}' not found.")
    
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
        raise ValueError(f"Inconsistent config list lengths: {{lengths}}")

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
        writer = csv.DictWriter(f, fieldnames=[{header_list}])
        writer.writeheader()
        for i in range(len(list(config.values())[0])):
            row = {{k: v[i] for k, v in config.items()}}
            row.update(fixed_values)  # Add fixed values
            writer.writerow(row)

    print(f"✅ {{args.output_file_path}}.csv written successfully.")
'''

    with open(output_file, "w") as f:
        f.write(script)

    print(f"📝 Generated '{output_file}' with headers: {headers} and fixed values: {fixed_values}")

def generate_writer_script_variable_sets(headers, fixed_values, variable_fields, output_file="write_job_config.py"):
    import_lines = [
        "import argparse",
        "import csv",
        "import os",
        "import json",
        "import itertools",
        "try:",
        "    import yaml",
        "except ImportError:",
        "    yaml = None  # YAML support is optional"
    ]

    all_fields = [h for h in headers if h not in fixed_values]
    config_block = "\n".join([f"    '{h}': []," for h in all_fields])
    header_list = ", ".join([f"'{h}'" for h in all_fields + variable_fields + list(fixed_values.keys())])
    
    fixed_block = "\n".join([f"    '{k}': '{v}'," for k, v in fixed_values.items()])

    script = f'''"""
Auto-generated config writer for job_configs.csv with fixed and variable fields.
Generates all combinations across all fields.
"""
{chr(10).join(import_lines)}


# === Option 1: Fill manually ===
config = {{
{config_block}
}}

# === Fixed values ===
fixed_values = {{
{fixed_block}
}}

variable_fields = {variable_fields}


# === Option 2: Load from JSON or YAML ===
def load_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{{file_path}}' not found.")
    
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
    for k, v in config.items():
        if not isinstance(v, list):
            raise ValueError(f"Config value for '{{k}}' must be a list")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, help="Path to write the CSV", default="job_configs.csv")
    parser.add_argument("--config_file", type=str, help="Path to a JSON/YAML config file")
    args = parser.parse_args()

    if args.config_file:
        config = load_from_file(args.config_file)
    
    validate_config(config)

    # Fields to combine (everything that's not fixed)
    all_config_fields = list(config.keys())

    # Generate Cartesian product
    all_combinations = list(itertools.product(*[config[k] for k in all_config_fields]))

    rows = []
    for values in all_combinations:
        row = dict(zip(all_config_fields, values))
        row.update(fixed_values)
        rows.append(row)

    # === Write to CSV ===
    with open(args.output_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[{header_list}])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✅ {{args.output_file_path}} written successfully.")
'''

    with open(output_file, "w") as f:
        f.write(script)

    print(f"📝 Generated '{output_file}' with headers: {headers}, fixed values: {fixed_values}, and variable fields: {variable_fields}")


def generate_writer_script_variable_sets_old(headers, fixed_values, variable_fields, output_file="write_job_config.py"):
    import_lines = [
        "import argparse",
        "import csv",
        "import os",
        "import json",
        "import itertools",
        "try:",
        "    import yaml",
        "except ImportError:",
        "    yaml = None  # YAML support is optional"
    ]

    # Identify parameter headers excluding fixed and variable fields
    param_headers = [h for h in headers if h not in fixed_values and h not in variable_fields]
    config_block = "\n".join([f"    '{h}': []," for h in param_headers + variable_fields])

    header_list = ", ".join([f"'{h}'" for h in variable_fields + param_headers + list(fixed_values.keys())])
    fixed_block = "\n".join([f"    '{k}': '{v}'," for k, v in fixed_values.items()])

    script = f'''"""
Auto-generated config writer for job_configs.csv with fixed and variable fields.
Generates all combinations of parameters per variable field value.
"""
{chr(10).join(import_lines)}


# === Option 1: Fill manually ===
config = {{
{config_block}
}}
# === Fixed values ===
fixed_values = {{
{fixed_block}
}}
variable_fields = {variable_fields}


# === Option 2: Load from JSON or YAML ===
def load_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{{file_path}}' not found.")
    
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
    for k, v in config.items():
        if not isinstance(v, list):
            raise ValueError(f"Config value for '{{k}}' must be a list")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", type=str, help="Path to write the CSV", default="job_configs.csv")
    parser.add_argument("--config_file", type=str, help="Path to a JSON/YAML config file")
    args = parser.parse_args()

    if args.config_file:
        config = load_from_file(args.config_file)
    
    validate_config(config)

    # Separate variable and param fields
    param_fields = [k for k in config if k not in variable_fields]
    variable_field_combinations = [config[vf] for vf in variable_fields]
    param_combinations = list(itertools.product(*[config[k] for k in param_fields]))

    rows = []
    for var_values in zip(*variable_field_combinations):
        for param_vals in param_combinations:
            row = dict(zip(variable_fields, var_values))
            row.update(dict(zip(param_fields, param_vals)))
            row.update(fixed_values)
            rows.append(row)

    # === Write to CSV ===
    with open(args.output_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[{header_list}])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✅ {{args.output_file_path}} written successfully.")
'''

    with open(output_file, "w") as f:
        f.write(script)

    print(f"📝 Generated '{output_file}' with headers: {headers}, fixed values: {fixed_values}, and variable fields: {variable_fields}")
