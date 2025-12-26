import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RLOOTrainer, RLOOConfig
from peft import LoraConfig, TaskType

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model_name = "EleutherAI/pythia-160m-deduped"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

rm_name = "distilbert-base-uncased-finetuned-sst-2-english"
reward_model = AutoModelForSequenceClassification.from_pretrained(rm_name, num_labels=2).to(device)

def build_dataset(tokenizer):
    ds = load_dataset("imdb", split="train[:1000]") 
    def format_prompts(examples):
        prompts = []
        for text in examples["text"]:
            prompt_text = " ".join(text.split()[:5]) 
            prompts.append(prompt_text)
        return {"prompt": prompts}
    
    ds = ds.map(format_prompts, batched=True, remove_columns=ds.column_names)
    return ds

train_dataset = build_dataset(tokenizer)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                
    lora_alpha=32,      
    lora_dropout=0.05,      
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)

rloo_config = RLOOConfig(
    output_dir="./rloo_imdb_lora",
    
    learning_rate=1.41e-4,             
    lr_scheduler_type="cosine",     
    warmup_ratio=0.05,              
    
    num_generations=4,         
    beta=0.05,                      
    
    bf16=False, fp16=False,        
    
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=8,  
    
    max_steps=200,              
    logging_steps=5,              
    
    gradient_checkpointing=True,   
    dataloader_pin_memory=False,
    max_completion_length=32,
)

trainer = RLOOTrainer(
    model=model,
    reward_funcs=reward_model,
    args=rloo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config, 
)

trainer.train()