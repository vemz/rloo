import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig  

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "EleutherAI/pythia-410m-deduped"
sft_adapter_path = "vemz/pythia-410m-sft-imdb"
output_dir = "./grpo_pythia_410m_explicit_pos"  

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map=None 
).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

model = PeftModel.from_pretrained(model, sft_adapter_path)
model = model.merge_and_unload()

rm_name = "lvwerra/distilbert-imdb"
reward_model = AutoModelForSequenceClassification.from_pretrained(rm_name, num_labels=2).to(device)
reward_tokenizer = AutoTokenizer.from_pretrained(rm_name)

def get_positive_score(prompts, completions, **kwargs):
    inputs = [p + c for p, c in zip(prompts, completions)]
    tokens = reward_tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = reward_model(**tokens)
    return outputs.logits[:, 1]

import re

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_dataset():
    ds = load_dataset("imdb", split="train[:300]")
    
    def format_prompts(examples):
        return {
            "prompt": [
                " ".join(clean_text(text).split()[:10])
                for text in examples["text"]
            ]
        }
        
    return ds.map(format_prompts, batched=True, remove_columns=ds.column_names)

def repetition_penalty_reward(completions):
    values = []
    for c in completions:
        words = c.lower().split()
        values.append(len(set(words)) / max(1, len(words)))
    return torch.tensor(values, device=device)

def combined_reward(prompts, completions, **kwargs):
    pos = get_positive_score(prompts, completions)
    rep = repetition_penalty_reward(completions)

    lengths = torch.tensor([len(c.split()) for c in completions], device=device)
    short_penalty = (lengths < 5).float() * -2.0
    
    return pos + 0.5 * rep + short_penalty

train_dataset = build_dataset()

rl_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)

grpo_config = GRPOConfig(
    output_dir=output_dir,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_generations=4,
    beta=0.05,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    max_steps=100,
    logging_steps=10,
    bf16=False,
    fp16=False,
    gradient_checkpointing=True,
    max_completion_length=48,
    save_strategy="steps",
    save_steps=40,
    save_total_limit=1,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=combined_reward, 
    args=grpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer, 
    peft_config=rl_peft_config, 
)

trainer.train()

trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")