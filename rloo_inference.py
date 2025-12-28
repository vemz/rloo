import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "EleutherAI/pythia-410m-deduped"
sft_path = "./sft_imdb_pythia_410m"          
rloo_path = "./rloo_pythia_410m_explicit_pos_3/final" 

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32 
).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = PeftModel.from_pretrained(model, sft_path)
model = model.merge_and_unload() 

model = PeftModel.from_pretrained(model, rloo_path)
model.to(device)
model.eval() 

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

prompts = [
    "The movie was", 
    "I honestly think that", 
    "At first I was bored, but then",
    "This film is a perfect example of",
    "The plot was",
    "The experiences I had while watching this movie were",
    "What a",
    "It is surprising how",
    "Absolutely",
]

for p in prompts:
    print(f" {p} \033[1m{generate(p)}\033[0m")
    print("-" * 60)