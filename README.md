# QLora fine-tuning LLM

## Install
In a conda environment with PyTorch / CUDA available, clone the repo and run in the top-level directory:
```bash
pip install -e .
```

## LLM Model
I use Llama-2-7b-hf as base model.
all detail are in config/base_config.yaml, you can modify it as you want.

## dataset
I use databricks-dolly-15k as dataset.

## Train
just run train.py and choose GPU, for example you can run:
```python
CUDA_VISIBLE_DEVICES=1 python train.py
```

## Evaluation
after training the model, you will get a new model saved in llama-7b-int4-dolly.
for example you can run:
```python
python evaluation.py --prompt "Write me a poem about Singapore." --max_new_tokens 128
```
