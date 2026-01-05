import torch
import os
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "EleutherAI/pythia-410m-deduped"
output_dir = "./sft_imdb_pythia_410m"

# Nettoyage
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# 2. Dataset
dataset = load_dataset("imdb", split="train[:2000]")

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important pour SFT

# 4. Modèle
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, 
    device_map=device
)

# 5. LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)

# 6. SFT Config (CORRIGÉE SELON LA DOC)
sft_config = SFTConfig(
    output_dir=output_dir,
    
    # Paramètres spécifiques SFT 
    dataset_text_field="text",  
    max_length=512,            
    packing=True,               
    
    # Paramètres d'entraînement classiques
    learning_rate=1e-4,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, 
    num_train_epochs=1,
    fp16=(device == "cuda"),
    logging_steps=10,
    save_strategy="epoch",
)

# 7. Trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,            # Tout passe par la config maintenant
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer, # Nouveau nom pour 'tokenizer'
)

print("Démarrage du SFT...")
trainer.train()

# Sauvegarde
print(f"Sauvegarde dans {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)