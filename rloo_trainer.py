import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, TaskType
from trl import RLOOTrainer, RLOOConfig

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "EleutherAI/pythia-410m-deduped"
sft_adapter_path = "./sft_imdb_pythia_410m"
output_dir = "./rloo_pythia_410m_explicit_pos"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32 
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
    tokens = reward_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = reward_model(**tokens)
    scores = outputs.logits[:, 1] 
    return scores

def build_dataset(tokenizer):
    ds = load_dataset("imdb", split="train[:1000]") 
    def format_prompts(examples):
        return {"prompt": [" ".join(text.split()[:5]) for text in examples["text"]]}
    ds = ds.map(format_prompts, batched=True, remove_columns=ds.column_names)
    return ds

train_dataset = build_dataset(tokenizer)

rl_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)

rloo_config = RLOOConfig(
    output_dir=output_dir,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_generations=4,
    beta=0.02,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    max_steps=200,
    logging_steps=10,
    bf16=False, fp16=False,
    gradient_checkpointing=True,
    max_completion_length=32,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
)

trainer = RLOOTrainer(
    model=model,
    reward_funcs=get_positive_score,
    args=rloo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=rl_peft_config, 
)

trainer.train()

trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")