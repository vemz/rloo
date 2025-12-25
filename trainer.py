import torch
import sys
import wandb
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RLOOTrainer, RLOOConfig
from datasets import load_dataset

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

wandb.init(project="rloo_debug_cpu_padding_fix")

DEVICE = "cpu" 

CURRENT_DIR = os.getcwd()
SFT_MODEL_PATH = os.path.join(CURRENT_DIR, "sft_model")      
RM_MODEL_PATH = os.path.join(CURRENT_DIR, "reward_model") 

rm_tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_PATH)

sft_tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

sft_tokenizer.padding_side = "left" 

sft_tokenizer.pad_token = sft_tokenizer.eos_token
if sft_tokenizer.eos_token_id is None:
    sft_tokenizer.eos_token = "<|endoftext|>"
    sft_tokenizer.pad_token = "<|endoftext|>"

rm_model = AutoModelForSequenceClassification.from_pretrained(RM_MODEL_PATH).to(DEVICE)
rm_model.eval() 

def custom_reward_func(prompts, completions, **kwargs):
    inputs_text = [p + c for p, c in zip(prompts, completions)]
    
    inputs = rm_tokenizer(
        inputs_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = rm_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        scores = probs[:, 1] 
        
    return scores.tolist()

def make_prompt(example):
    words = example["text"].split()
    prompt_len = min(len(words), 5)
    return {"prompt": " ".join(words[:prompt_len])}

dataset = load_dataset("stanfordnlp/imdb", split="test")
train_dataset = dataset.select(range(50)).map(make_prompt)

rloo_config = RLOOConfig(
    output_dir="./resultats_rloo_final",
    run_name="rloo_cpu_fix",
    
    num_generations=4,          
    temperature=0.7,            
    beta=0.05,
    
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=2, 
    
    learning_rate=1e-5, 
    max_grad_norm=1.0,  
    
    max_prompt_length=64,       
    max_completion_length=64,   
    
    max_steps=10,     
    logging_steps=1,            
    save_strategy="no",
    report_to="wandb",
    
    fp16=False,                 
    bf16=False,
    remove_unused_columns=False,
)

trainer = RLOOTrainer(
    model=SFT_MODEL_PATH,
    reward_funcs=custom_reward_func,
    args=rloo_config,
    train_dataset=train_dataset,
    processing_class=sft_tokenizer, 
)

trainer.train()
trainer.save_model("./rloo_final_model2")