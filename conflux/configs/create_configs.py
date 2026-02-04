import sys
from configs import generate_writer_script, generate_writer_script_fixed, generate_writer_script_variable_sets

# TODO: The only downside is if I want to repeat a certain value, it has to be modified in the automatically generated script. For example, if I want to repeat the same dataset_name for multiple model_type and temperature combinations, I have to modify the script accordingly.

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_configs.py --headers header1 header2 header3 ... --fixed_values key1=value1 key2=value2 ... [--variable_fields variable_field1 variable_field2 ...] [output_file_path]")
        #print("Usage: python create_configs.py --headers header1 header2 header3 ... --fixed_values key1=value1 key2=value2 ... [output_file_path]")
        print("Example: python create_configs.py --headers dataset_name model_type temperature --fixed_values fixed_key=fixed_value --variable_fields variable_field1 variable_field2 --output_file_path write_job_config.py")
        #print("Example: python create_configs.py --headers dataset_name model_type temperature --fixed_values fixed_key=fixed_value --output_file_path write_job_config.py")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Generate a config writer script.")
    parser.add_argument("--headers", nargs='+', required=True, help="List of headers for the config CSV.")
    parser.add_argument("--fixed_values", nargs='*', default=[], help="List of fixed values in the format key=value.")
    parser.add_argument("--variable_fields", nargs='*', default=[], help="List of variable fields to generate combinations for.")
    parser.add_argument("--output_file_path", type=str, default="write_job_config.py", help="Output file for the generated script.")
    args = parser.parse_args()
    headers = args.headers
    fixed_values = {}
    for item in args.fixed_values:
        if '=' in item:
            key, value = item.split('=', 1)
            fixed_values[key] = value
        else:
            print(f"Warning: '{item}' is not in key=value format and will be ignored.")
    output_file_path = args.output_file_path
    if args.variable_fields and fixed_values:
        variable_fields = args.variable_fields
        generate_writer_script_variable_sets(headers, fixed_values, variable_fields, output_file_path)
    elif fixed_values:
        generate_writer_script_fixed(headers, fixed_values, output_file_path)
    else:
        generate_writer_script(headers, output_file)
    