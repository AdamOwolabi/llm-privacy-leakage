"""
Models for LLM drift experiments.
download setup and rumnner models
Usage:
    python download_models.py
"""

import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Free gated models (require HF token, but free to use)
MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-2b": "google/gemma-2-2b-it",
    #"gemma-2-9b": "google/gemma-2-9b-it",
}

DEFAULT_MODEL = "llama-3.1-8b"  # for testing


def authenticate():
    """Login with HuggingFace token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("env.")
    login(token=token)


def load_model(model_key=DEFAULT_MODEL):
    """Load model and tokenizer."""
    authenticate()
    model_id = MODELS.get(model_key, model_key)
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer


def query(prompt, model_key=DEFAULT_MODEL):
    """Generate response from model."""
    model, tokenizer = load_model(model_key)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    response = query("Hello, how are you?")
    print(response)

