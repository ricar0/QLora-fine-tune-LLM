from transformers import AutoTokenizer

from config import configs

# load tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained(configs.base_model.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"