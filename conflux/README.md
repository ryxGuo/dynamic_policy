
# **Conflux**
Generates job configuration CSV files for experiments, training jobs, or batch runs. Can create configs using pre-defined **variable parameters** and **fixed values**, or load configurations from JSON or YAML files.


## 🚀 Usage

### 1. Define configs in the script and generate the relevant python file

In the generated file: e.g: `create_train_config.py`:

```python
# Command to create the python file that can generate your configurations
#
# Parameters:
#   --headers: List of all configuration keys that do not remain constant/fixed.
#   --fixed_values: List of parameters, the values for which stay constant for all configurations
#   --variable_fields: List of fields for which we generate all possible configurations.
#                      For e.g: We may have a set of models in our header_values, and for each model, we want to train it with 3 different learning rates.
#                      Therefore, the variable_field would be learning_rate in this case. An example is provided below for reference.
#   --output_file_path: Path at which the generated python file is saved.


python create_configs.py --headers header1 header2 \
                         --fixed_values key1=value1 key2=value2 \
                         --variable_fields variable_field1 variable_field2 \
                         --output_file_path <output_file_path_name>
```

Example usage:

```python
python create_configs.py --headers dataset_name gas --fixed_values model=gpt2 epsilon=4 --variable_fields learning_rate --output_file_path create_train_configs.py
```

```python
config = {
    "dataset_name": ["cifar10", "mnist"],
    "learning_rate": [1e-3, 1e-4],
    "gas": [1, 16]
}

fixed_values = {
    "model": "cnn",
    "epsilon": 4
}

variable_fields = ["learning_rate"]
```

### 2. Run the generated python file to create a csv with the configs you want

You will get an python file generated at output_file_path (say, create_train_configs.py) which you can then use

```python
# Command to run the python file that will generate your experimental configuration csv.
#
# Parameters:
#   --output_file_path: Path at which the generated exp configuration csv is saved.
#   --config_file: Path from where the yaml/json config file values are loaded (if it is available)

python create_train_configs.py --output_file_path <output_file_path> \
                               --config_file <config_file_path, Optional>
```

Example  output

```python
dataset,lr,gas,model,epsilon
cifar10,1e-3,1,cnn,4
cifar10,1e-4,1,cnn,4
mnist,1e-3,16,cnn,4
mnist,1e-4,16,cnn,4
```

