import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

pd.set_option('display.max_colwidth', None)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("stanfordnlp/imdb")

dataset_split = ds['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split['train'] 
rloo_dataset = dataset_split['test']  

# reward model 

rm_model_name = "distilbert-base-uncased"
tokenizer_rm = AutoTokenizer.from_pretrained(rm_model_name)

def preprocess_rm(examples):
    texts = [t.replace("<br />", " ") for t in examples["text"]]
    # truncate at max length 512 for DistilBERT
    return tokenizer_rm(texts, truncation=True, padding="max_length", max_length=512)

tokenized_rm_train = train_dataset.map(preprocess_rm, batched=True)
tokenized_rm_eval = rloo_dataset.select(range(500)).map(preprocess_rm, batched=True)

model_rm = AutoModelForSequenceClassification.from_pretrained(rm_model_name, num_labels=2)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# training rm 
training_args_rm = TrainingArguments(
    output_dir="./resultats_rm",
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="steps", 
    eval_steps=500,
    save_strategy="no",
    logging_steps=100,
    use_mps_device=(DEVICE == "mps"), 
    report_to="none" 
)

trainer_rm = Trainer(
    model=model_rm,
    args=training_args_rm,
    train_dataset=tokenized_rm_train,
    eval_dataset=tokenized_rm_eval,
    tokenizer=tokenizer_rm,
    compute_metrics=compute_metrics,
)

trainer_rm.train()

trainer_rm.save_model("./reward_model")
tokenizer_rm.save_pretrained("./reward_model")

del model_rm, trainer_rm
torch.cuda.empty_cache() if torch.cuda.is_available() else None

rm_check = AutoModelForSequenceClassification.from_pretrained("./reward_model").to(DEVICE)
print(f"weight type: {next(rm_check.parameters()).dtype}")

test_texts = ["This movie was a masterpiece!", "I hated this film, waste of time."]
inputs = tokenizer_rm(test_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

with torch.no_grad():
    outputs = rm_check(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)

print(f"positive sentence ('{test_texts[0]}') proba positive : {probs[0][1]:.4f}")
print(f"negative sentence ('{test_texts[1]}') proba positive : {probs[1][1]:.4f}")

if probs[0][1] > probs[1][1]:
    print("working")
else:
    print("not working")
del rm_check


# sft policy model

policy_model_name = "distilgpt2"
tokenizer_policy = AutoTokenizer.from_pretrained(policy_model_name)
tokenizer_policy.pad_token = tokenizer_policy.eos_token

def preprocess_sft(examples):
    return tokenizer_policy(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_sft_train = train_dataset.map(preprocess_sft, batched=True)

model_policy = AutoModelForCausalLM.from_pretrained(policy_model_name)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_policy, mlm=False)

training_args_sft = TrainingArguments(
    output_dir="./resultats_sft",
    learning_rate=2e-5,
    per_device_train_batch_size=8, 
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="no",
    use_mps_device=(DEVICE == "mps"),
    report_to="none"
)

trainer_sft = Trainer(
    model=model_policy,
    args=training_args_sft,
    train_dataset=tokenized_sft_train,
    tokenizer=tokenizer_policy,
    data_collator=data_collator,
)

trainer_sft.train()

trainer_sft.save_model("./sft_model")
tokenizer_policy.save_pretrained("./sft_model")

del model_policy, trainer_sft
torch.cuda.empty_cache() if torch.cuda.is_available() else None

policy_check = AutoModelForCausalLM.from_pretrained("./sft_model").to(DEVICE)
print(f"weight type (sft) : {next(policy_check.parameters()).dtype}")

seed_text = "The movie was"
inputs_sft = tokenizer_policy(seed_text, return_tensors="pt").to(DEVICE)

print(f"generating from : '{seed_text}'...")
gen_tokens = policy_check.generate(
    **inputs_sft, 
    max_new_tokens=30, 
    do_sample=True, 
    temperature=0.7,
    pad_token_id=tokenizer_policy.eos_token_id
)
gen_text = tokenizer_policy.decode(gen_tokens[0], skip_special_tokens=True)
print(f"result: {gen_text}")