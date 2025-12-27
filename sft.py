import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model_name = "EleutherAI/pythia-410m-deduped" 
output_dir = "./sft_imdb_pythia_410m"

dataset = load_dataset("imdb", split="train[:2000]")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, 
).to(device)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)

sft_config = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    max_length=512,
    packing=True,            
    learning_rate=1e-4,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, 
    num_train_epochs=1,
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 6. Trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)