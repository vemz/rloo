import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cpu"

SFT_PATH = "./sft_model"
RLOO_PATH = "./rloo_final_model"

tokenizer = AutoTokenizer.from_pretrained(SFT_PATH)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

model_sft = AutoModelForCausalLM.from_pretrained(SFT_PATH).to(DEVICE)
model_sft.eval()

model_rloo = AutoModelForCausalLM.from_pretrained(RLOO_PATH).to(DEVICE)
model_rloo.eval()

prompt_text = "The movie was"
inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

print("before")
for i in range(3):
    gen = model_sft.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.8)
    print(f"v{i+1}: {tokenizer.decode(gen[0], skip_special_tokens=True)}")

print("after")
for i in range(3):
    gen = model_rloo.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.8)
    print(f"v{i+1}: {tokenizer.decode(gen[0], skip_special_tokens=True)}")