from transformers import AutoModelForCausalLM,BitsAndBytesConfig, GenerationConfig
import torch
from config import configs
from peft import PeftModel
from model.base_model import LLama
import argparse
model_id = configs.base_model.model_name
peft_path = "./llama-7b-int4-dolly/checkpoint-200"

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='Write me a poem about Singapore.')
parser.add_argument('--max_new_tokens', type=int, default=128)
args = parser.parse_args()
# loading model
bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto"
)

# loading peft weight
model = PeftModel.from_pretrained(
    model,
    peft_path,
    torch_dtype=torch.float16,
)
model.eval()

# generation config
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4, # beam search
)

llama = LLama()

with torch.no_grad():
    prompt = args.prompt
    inputs = llama.tokenizer(prompt, return_tensors="pt").to('cuda')
    generation_output = model.generate(
        input_ids=inputs.input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=args.max_new_tokens,
    )
    print('\nAnswer: ', llama.tokenizer.decode(generation_output.sequences[0]))

