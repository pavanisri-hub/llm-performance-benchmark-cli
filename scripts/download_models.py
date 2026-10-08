"""Pre-download Hugging Face models for benchmarking."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    "distilgpt2",
    "gpt2",
    "EleutherAI/pythia-160m",
]

for model_id in models:
    print(f"Downloading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    print(f"{model_id} downloaded successfully.")

print("All models downloaded.")