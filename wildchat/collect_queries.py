from pathlib import Path
from datasets import load_dataset
from judge_model import JudgeModel
import os
import json

WILDCHAT_PATH = "allenai/WildChat"
PATH_TO_PROMPT_DIR = "/home/kramesh3/projects/dynamic_policy/wildchat/prompts/"

def load_dataset_and_preprocess(path_to_dataset):
    dataset = load_dataset(path_to_dataset)
    return dataset

def read_prompt_from_text_file(path_to_text_file):
    with open(path_to_text_file, 'r') as file:
        prompt_str = file.read()
    return prompt_str


if __name__ == "__main__":
    version = "version-main"
    prompt_style = "style2"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    index_start, index_end = 0, 30

    ds = load_dataset_and_preprocess(WILDCHAT_PATH)
    single_prompt = read_prompt_from_text_file(f"{PATH_TO_PROMPT_DIR}/{prompt_style}.txt")
    
    #model_name = "allenai/OLMo-2-1124-7B-Instruct"
    judge = JudgeModel(model_name = model_name)


    output_dir = Path("./wildchat_queries")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / f"privacy_risks-{prompt_style}-{version}-{index_start}-{index_end}.json"
    privacy_risks, prompt_list = [], []

    
    for i in range(index_start, index_end):
        if ds['train'][i]['language'] != "English":
            continue
        prompt_test = ds['train'][i]['conversation'][0]['content']
        prompt = single_prompt.format(prompt=prompt_test)
        prompt_list.append(prompt)
        
    full_texts, model_answers = judge.judge(prompt_list, max_length=50, batch_size = 8) 
    
    # Saving the prompts that are privacy risks
    for prompt_original, full_text, model_answer in zip(prompt_list, full_texts, model_answers):  
        if ds['train'][i]['language'] != "English":
            continue
        # prompt_original = ds['train'][i]['conversation'][0]['content']
        if "Privacy Risk Level: YES" in model_answer:
            
            # Store the data
            entry = {
                "prompt": prompt_original,
                "model_answer": model_answer,
                "full_text": full_text,
                "privacy_risk": True
            }
        else:
            entry = {
                "prompt": prompt_original,
                "model_answer": model_answer,
                "full_text": full_text,
                "privacy_risk": False
            }
        
        privacy_risks.append(entry)

    # Save as JSON
    with open(json_file, 'w') as f:
        json.dump(privacy_risks, f, indent=2)

    print(f"\n Saved {len(privacy_risks)} analyzed privacy risk queries to {json_file}")