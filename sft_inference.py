import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "EleutherAI/pythia-410m-deduped"
adapter_path = "./sft_imdb_pythia_410m"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = PeftModel.from_pretrained(base_model, adapter_path)
model.to(device)
model.eval()

def test_model(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with model.disable_adapter():
        with torch.no_grad():
            outputs_base = model.generate(
                **inputs, 
                max_new_tokens=50,        
                do_sample=True,        
                temperature=0.7,        
                top_k=50,               
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

    with torch.no_grad():
        outputs_sft = model.generate(
            **inputs, 
            max_new_tokens=50,        
            do_sample=True,        
            temperature=0.7,        
            top_k=50,               
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    base_text = tokenizer.decode(outputs_base[0], skip_special_tokens=True)[len(prompt_text):]
    sft_text = tokenizer.decode(outputs_sft[0], skip_special_tokens=True)[len(prompt_text):]
    
    print(f"Prompt : {prompt_text}")
    print(f"Base   : {base_text}")
    print(f"SFT    : {sft_text}")
    print("-" * 60)

prompts = [
    "The movie was",                 
    "I honestly think that",         
    "The acting in this film is",      
    "I went to the cinema and",         
    "At first I was bored, but then",
    "The dog in this",
    "What is amazing",  
]

for p in prompts:
    test_model(p)