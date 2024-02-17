from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from config import configs
from model.base_model import LLama
from utils.my_dataset import MyDataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
def parse_args():
    args = TrainingArguments(
        output_dir=configs.train_config.output_dir,
        num_train_epochs=configs.train_config.num_train_epochs,
        max_steps=configs.train_config.max_steps,
        fp16=configs.train_config.fp16,
        optim=configs.train_config.optim,
        learning_rate=configs.train_config.learning_rate,
        lr_scheduler_type=configs.train_config.lr_scheduler_type,
        per_device_train_batch_size=configs.train_config.per_device_train_batch_size,
        gradient_accumulation_steps=configs.train_config.gradient_accumulation_steps,
        gradient_checkpointing=configs.train_config.gradient_checkpointing,
        group_by_length=configs.train_config.group_by_length,
        logging_steps=configs.train_config.logging_steps,
        save_strategy=configs.train_config.save_strategy,
        save_total_limit=configs.train_config.save_total_limit,
        disable_tqdm=configs.train_config.disable_tqdm,
        remove_unused_columns=False,
    )
    return args

def get_data(test_size):
    mydataset = MyDataset(configs.dataset)
    train_data, val_data = mydataset.split_data(test_size, True, 42)
    return train_data, val_data

def lora_model(model):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=configs.lora_model.r,
        lora_alpha=configs.lora_model.lora_alpha,
        lora_dropout=configs.lora_model.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias=configs.lora_model.bias,
        task_type=configs.lora_model.task_type
    ) 
    model = get_peft_model(model, peft_config)
    return model
if __name__ == '__main__':
    
    print("======loading args======")
    args = parse_args()
    
    print("======loading model======")
    llama = LLama()
    model = lora_model(llama.model)
    
    print("======loading dataset======")
    test_size = 1000
    train_data, val_data = get_data(test_size)
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=args,
        data_collator=DataCollatorForSeq2Seq(
            llama.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )
    # silence the warnings. re-enable for inference!
    llama.model.config.use_cache = False
    print("======start training======")
    trainer.train()
    print("======end training======")
    llama.model.save_pretrained("llama-7b-int4-dolly")