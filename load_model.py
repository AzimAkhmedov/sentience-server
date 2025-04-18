import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
model_path = "./saved_therapist_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

# Generation function
def get_response(prompt, temperature=0.7, top_p=0.9, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)

    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    return full_output.strip()