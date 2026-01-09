import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login()

device = "mps" 
base_model_id = "EleutherAI/pythia-410m-deduped"
local_model_path = "./rloo_pythia_410m_explicit_pos_3/final" 
repo_name = "vemz/pythia-410m-rloo-imdb" 

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32).to(device)

model = PeftModel.from_pretrained(base_model, local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)