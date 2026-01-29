import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import List, Union

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class JudgeModel:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    def _format_prompt(self, prompt: str) -> str:
        """Format a single prompt with chat template if available."""
        
        if self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "You are an expert judge."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        return formatted_prompt
    
    def judge(self, prompt: Union[str, List[str]], max_length: int = 512, batch_size: int = 1) -> Union[str, List[str]]:
        """
        Inference pipeline - supports single prompt or batch of prompts.
        
        Args:
            prompt: Single prompt (str) or list of prompts (List[str])
            max_length: Maximum length of generated response
        
        Returns:
            Single response (str) or list of responses (List[str])
        """
        
        # Handle single prompt
        if isinstance(prompt, str):
            return self._judge_single(prompt, max_length)
        
        # Handle batch of prompts
        elif isinstance(prompt, list):
            return self._judge_batch(prompt, max_length, batch_size)
        
        else:
            raise TypeError("prompt must be str or List[str]")
    
    def _judge_single(self, prompt: str, max_length: int) -> str:
        """Process a single prompt."""
        
        if self.tokenizer.chat_template is not None:
            print("Using chat template...")
        else:
            print("No chat template found, using the normal prompt.")
        
        formatted_prompt = self._format_prompt(prompt)
        
        with torch.no_grad():
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            outputs = self.model.generate(input_ids=inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"],
                                              max_new_tokens=max_length,
                                              do_sample=True, top_p=0.95, top_k=50, temperature=0.7,
                                              pad_token_id=self.model.config.pad_token_id,
                                              eos_token_id=self.model.config.eos_token_id,
                                              use_cache=True)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_len = inputs["attention_mask"].sum(dim=1)
        new_tokens_only = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return response, new_tokens_only
    
    def _judge_batch(self, prompts: List[str], max_length: int, batch_size: int) -> List[str]:
        """Process a batch of prompts efficiently."""
        
        print(f"Processing batch of {len(prompts)} prompts...")
        
        if self.tokenizer.chat_template is not None:
            print("Using chat template for all prompts...")
        else:
            print("No chat template found, using normal prompts...")
        
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        responses = []
        new_tokens_only = []

        with torch.no_grad():
            for i in range(0, len(formatted_prompts), batch_size):
                batch_prompts = formatted_prompts[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                input_lengths = inputs["attention_mask"].sum(dim=1)
                
                # Generate for batch
                outputs = self.model.generate(input_ids=inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"],
                                              max_new_tokens=max_length,
                                              do_sample=True, top_p=0.95, top_k=50, temperature=0.7,
                                              pad_token_id=self.model.config.pad_token_id,
                                              eos_token_id=self.model.config.eos_token_id,
                                              use_cache=True)
        
                # Decode batch
                responses.extend([
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ])
                
                new_tokens_only.extend([
                    self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
                    for output, input_len in zip(outputs, input_lengths)
                ])
    
        return responses, new_tokens_only
        

if __name__ == "__main__":
    judge = JudgeModel(model_name = "Qwen/Qwen2.5-7B-Instruct")
    
    # Example 1: Single prompt
    print("=" * 60)
    print("SINGLE PROMPT TEST")
    print("=" * 60)
    single_prompt = """Evaluate this response:
Response: "The capital of France is Paris."
Provide: 1) Score 1-10, 2) Brief feedback"""
    
    result = judge.judge(single_prompt, max_length=256)
    print("Judge's Evaluation:")
    print(result)
    
    # Example 2: Batch of prompts
    print("\n" + "=" * 60)
    print("BATCH PROMPTS TEST")
    print("=" * 60)
    batch_prompts = [
        """Evaluate: "The capital of France is Paris."
Score 1-10.""",
        """Evaluate: "2 + 2 = 4"
Score 1-10.""",
        """Evaluate: "The Earth is flat."
Score 1-10.""",
    ]
    
    results = judge.judge(batch_prompts, max_length=10)
    print("Batch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Response {i} ---")
        print(result)