from pathlib import Path
from datasets import load_dataset
from judge_model import JudgeModel
import os, re, json
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

WILDCHAT_PATH = "allenai/WildChat"
PATH_TO_PROMPT_DIR = "/home/kramesh3/projects/dynamic_policy/wildchat/prompts/"

def load_dataset_and_preprocess(path_to_dataset):
    dataset = load_dataset(path_to_dataset)
    return dataset

def read_prompt_from_text_file(path_to_text_file):
    with open(path_to_text_file, 'r') as file:
        prompt_str = file.read()
    return prompt_str


@dataclass
class ScriptArguments:
    """Arguments for the privacy risk analysis script"""
    
    version: str = field(
        default="version-main",
        metadata={"help": "Version identifier for the experiment"}
    )
    prompt_style: str = field(
        default="style1_revised",
        metadata={"help": "Prompt style to use"}
    )
    model_name: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Name of the model to use for judging"}
    )
    index_start: int = field(
        default=0,
        metadata={"help": "Starting index for dataset samples"}
    )
    index_end: int = field(
        default=50,
        metadata={"help": "Ending index for dataset samples"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for model inference"}
    )
    max_length: int = field(
        default=150,
        metadata={"help": "Maximum length for model outputs"}
    )
    output_dir: str = field(
        default="./wildchat_queries",
        metadata={"help": "Output directory for results"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    version = args.version
    prompt_style = args.prompt_style
    model_name = args.model_name
    index_start, index_end = args.index_start, args.index_end
    batch_size = args.batch_size
    max_length = args.max_length

    ds = load_dataset_and_preprocess(WILDCHAT_PATH)
    single_prompt = read_prompt_from_text_file(f"{PATH_TO_PROMPT_DIR}/{prompt_style}.txt")
    
    judge = JudgeModel(model_name=model_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_replace = model_name.replace('/', '_')
    json_file = f"privacy_risks-{prompt_style}-{version}-{index_start}-{index_end}-model-{model_name_replace}.json"
    privacy_violations, non_privacy_violations, prompt_list, prompt_original_list = [], [], [], []

    
    for i in range(index_start, index_end):
        if ds['train'][i]['language'] != "English":
            continue
        prompt_test = ds['train'][i]['conversation'][0]['content']
        prompt = single_prompt.format(prompt=prompt_test)
        prompt_list.append(prompt)
        prompt_original_list.append(prompt_test)
        
    full_texts, model_answers = judge.judge(prompt_list, max_length=max_length, batch_size=batch_size) 
    
    # Saving the prompts that are privacy risks
    for prompt_original, full_text, model_answer in zip(prompt_original_list, full_texts, model_answers):
        if re.search(r'privacy\s*.*?\s*risk\s*.*?\s*level\s*.*?\s*yes', model_answer, re.IGNORECASE):
            entry = {
                "prompt": prompt_original,
                "model_answer": model_answer,
                "full_text": full_text,
                "privacy_risk": True
            }
            privacy_violations.append(entry)
        else:
            entry = {
                "prompt": prompt_original,
                "model_answer": model_answer,
                "full_text": full_text,
                "privacy_risk": False
            }
            non_privacy_violations.append(entry)

    # Save as JSON
    with open(f'{output_dir}/all-{json_file}', 'w') as f:
        json.dump(privacy_violations + non_privacy_violations, f, indent=2)
    
    with open(f'{output_dir}/PV-{json_file}', 'w') as f:
        json.dump(privacy_violations, f, indent=2)
    
    with open(f'{output_dir}/NPV-{json_file}', 'w') as f:
        json.dump(non_privacy_violations, f, indent=2)

    print(f"\n Saved {len(privacy_violations + non_privacy_violations)} analyzed privacy risk queries to {json_file}")