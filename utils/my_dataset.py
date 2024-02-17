import datasets
import sys
import os
from datasets import load_dataset
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from prompt.base_prompt import prompt_template
from config import configs
from utils import tokenizer

def generate_prompt(instruction, input=None, label=None, prompt_template=prompt_template):
    if input:
        res = prompt_template["prompt_input"].format(
            instruction=instruction,
            input=input
        )
    else:
        res = prompt_template["prompt_no_input"].format(
            instruction=instruction
        )
    if label:
        res = f"{res}{label}"
    return res

def tokenize(tokenizer, prompt, max_length=configs.base_model.max_length, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(
        data_point["instruction"],
        data_point["context"],
        data_point["response"],
    )
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)
    user_prompt = generate_prompt(data_point["instruction"], data_point["context"])
    tokenized_user_prompt = tokenize(tokenizer, user_prompt)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    mask_token = [-100]*user_prompt_len
    tokenized_full_prompt["labels"] = mask_token+tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt

class MyDataset(Dataset):
    def __init__(self, config_dataset):
        self.dataset = datasets.load_dataset(
            path=config_dataset.name,
            split=config_dataset.split
        )
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    def split_data(self, test_size, shuffle, seed):
        self.dataset = self.dataset.train_test_split(test_size=test_size, shuffle=shuffle, seed=seed)
        cols = ["instruction", "context", "response", "category"]
        train_data = self.dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
        val_data = self.dataset["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
        return train_data, val_data


