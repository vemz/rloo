import os
import shutil
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType
from trl import ModelConfig, ScriptArguments
# ON IMPORTE BIEN L'EXPERIMENTAL COMME DEMANDÉ
from trl.experimental.ppo import PPOConfig, PPOTrainer

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. CONFIGURATION ---
# On définit les arguments manuellement pour le notebook
# Config Reward Model
reward_model_args = ModelConfig(
    model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english",
    trust_remote_code=True,
)
# Config Modèle
model_args = ModelConfig(
    model_name_or_path = "EleutherAI/pythia-160m-deduped",
    trust_remote_code=True,
)

# Config Script
script_args = ScriptArguments(
    dataset_name="imdb",
    dataset_train_split="train",
)




# Config PPO (Experimental)
# Note : PPO charge 4 modèles en VRAM. On doit être très économe.
training_args = PPOConfig(
    output_dir="ppo-imdb",
    run_name="ppo-imdb",

    # Hyperparamètres d'apprentissage
    learning_rate=1e-6,             # LR très bas pour la stabilité
    per_device_train_batch_size=1,  # Batch de 1 OBLIGATOIRE sur Colab (sinon OOM)
    gradient_accumulation_steps=16, # On compense avec l'accumulation
    num_ppo_epochs=2,
    num_mini_batches=1,

    # Paramètres PPO
    total_episodes=500,             # Limité pour le test
    stop_token="eos",               # Important pour Qwen    #   ON NE SAIT PAS SI C'est important dans notre cas
    missing_eos_penalty=1.0,

    # Hardware
    fp16=True,
    seed=42,
)

# Nettoyage
if os.path.exists(training_args.output_dir):
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

# --- 2. CONFIGURATION LoRA (Crucial pour la mémoire) ---
# On applique LoRA à la Policy pour qu'elle prenne moins de place
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)


# --- 3. MODEL & TOKENIZER ---
print("--- Chargement Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    padding_side="left",            # Toujours gauche pour PPO/Génération
    trust_remote_code=model_args.trust_remote_code
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("--- Chargement des 4 Modèles PPO ---")

# A. Policy (L'Acteur)
policy = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    #torch_dtype=torch.float16,
    device_map="auto"
)

# B. Ref Policy (Référence - Gelé)
ref_policy = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    #torch_dtype=torch.float16,
    device_map="auto"
)
ref_policy.eval()

# C. Reward Model (Le Juge)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_args.model_name_or_path,
    num_labels=1,
    #torch_dtype=torch.float16,
    device_map="auto"
)
reward_model.config.pad_token_id = tokenizer.pad_token_id

# D. Value Model (Le Critique)   # A REVOIR                               #  CE QU'IL Y A EN PLUS DANS LE PPO
value_model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    num_labels=1,
    #torch_dtype=torch.float16,
    device_map="auto"
)
value_model.config.pad_token_id = tokenizer.pad_token_id

# --- 4. DATASET ---
print("--- Préparation du Dataset ---")

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


# --- 5. INITIALISATION DU TRAINER  ---
print("--- Initialisation PPOTrainer  ---")

# On utilise DataCollatorWithPadding pour être sûr que les batchs sont bien formés
data_collator = DataCollatorWithPadding(tokenizer)

trainer = PPOTrainer(
    args=training_args,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    data_collator=data_collator,
    peft_config=peft_config, # LoRA appliqué à la Policy
)

# --- 6. LANCEMENT ---
print("Démarrage de l'entraînement PPO...")
trainer.train()

# Sauvegarde
trainer.save_model(training_args.output_dir)
print("Terminé ! Modèle PPO sauvegardé.")