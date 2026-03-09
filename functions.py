"""
queries llm for synthetic outputs, saves output to database 

Usage:
    python functions.py
"""
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv
load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found. Put it in .env")
login(token=hf_token)

def greet_user(name):
    """This function prints a personalized greeting."""
    print(f"Hello, {name}! Great to see you.")


def query_llm(domain, prompt, model, hidden_trait="Spending Habit"):
    """
    Generate synthetic outputs for high/medium/low in one run.
    Returns compact structured JSON (no input/system_prompt fields).
    """
    level_behaviors = {
        "high": "always seeking deals, tracks spending meticulously",
        "medium": "budget conscious, generally tracks spending",
        "low": "uses the occasional coupons",
    }

    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    outputs = []
    for level, behavior in level_behaviors.items():
        system_prompt = (
            f"You are generating synthetic output for domain '{domain}'. "
            f"Hidden Trait: {hidden_trait}. Sensitivity: {level}. "
            f"Behavior profile: {behavior}. "
            "Do not reveal the hidden trait directly."
        )
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model_obj.device)

        with torch.no_grad():
            output_tokens = model_obj.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response_text = decoded.split("Assistant:")[-1].strip()

        outputs.append(
            {
                "sensitivity_level": level,
                "behavior": behavior,
                "response": response_text,
            }
        )

    return {
        "domain": domain,
        "model": model,
        "hidden_trait": hidden_trait,
        "outputs": outputs,
    }


def generate_personalized_output(model_name, hidden_trait, sensitivity_level, prompt_sequence):
    """
    Generates synthetic outputs conditioned on a hidden user trait and sensitivity level.
    
    Args:
        model_name (str): Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-chat-hf').
        hidden_trait (str): The specific user attribute (e.g., 'income level').
        sensitivity_level (str): 'Low', 'Medium', 'High', or 'Control'.
        prompt_sequence (list): List of user prompts in the conversation.
        
    Returns:
        str: The LLM's generated response.
    """
    # 1. Load the open-weight model and tokenizer from Hugging Face 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # 2. Implement Personalization Layer (Memory/Context Conditioning) 
    # We embed the trait and sensitivity directly into the system context to "condition" the model.
    if sensitivity_level.lower() == "control":
        system_context = "You are a helpful shopping assistant."
    else:
        system_context = f"You are a shopping assistant. Your user has a {sensitivity_level} level of {hidden_trait}. Adapt your tone and recommendations accordingly."

    # 3. Format the conversation history (Memory)
    # Concatenating prompts to simulate personalization drift over time 
    full_conversation = system_context + "\n" + "\n".join(prompt_sequence)
    
    inputs = tokenizer(full_conversation, return_tensors="pt").to(model.device)

    # 4. Generate the output (Generation-only mode) 
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Return only the new response part
    return response.split(prompt_sequence[-1])[-1].strip()


def export_json():
    pass


if __name__ == "__main__":
    print(greet_user("Alex")) 
    model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    result = query_llm(
        domain="shopping",
        prompt="I need to reduce my weekly grocery spending. What should I do?",
        model=model_id,
    )
    print(resulta)
    #print(json.dumps(result, indent=2))
    
