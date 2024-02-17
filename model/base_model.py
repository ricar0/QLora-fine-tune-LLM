from pydantic import BaseModel
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from config import configs

class LLama():
    def __init__(self, load_in_4bit=True):
        self.model_id = configs.base_model.model_name

        # load tokenizer from huggingface
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        bnb_config = None 
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # load model from huggingface
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=configs.base_model.device_map,
            use_cache=False
        )

    def run_llama(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generate_ids = self.model.generate(input_ids.to("cuda"), max_new_tokens=64)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        return response

if __name__ == '__main__':
    llama = LLama()
    print(llama.run_llama("what can computer do?"))



        