import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION (Identique au RLOO) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

# Paramètres Modèles
BASE_MODEL_NAME = "EleutherAI/pythia-410m-deduped"
SFT_ADAPTER_PATH = "vemz/pythia-410m-sft-imdb" # Le point de départ
REWARD_MODEL_NAME = "lvwerra/distilbert-imdb"  # Le Juge

# Hyperparamètres d'entraînement (Alignés sur RLOO)
LEARNING_RATE = 3e-5
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1
KL_BETA = 0.02             # Le "beta" du RLOO
BATCH_SIZE = 2             # per_device_train_batch_size
GRAD_ACCUMULATION = 16     # gradient_accumulation_steps
MAX_STEPS = 200            # max_steps
MAX_NEW_TOKENS = 32        # max_completion_length

# Paramètres spécifiques PPO (Que RLOO n'a pas)
PPO_EPOCHS = 4             # Nombre de passes d'opti sur un batch généré
CLIP_EPS = 0.2             # Standard PPO Clipping
VALUE_LOSS_COEF = 0.5      # Poids de la perte du critique

# --- 2. PRÉPARATION DU DATASET ---
# Exactement la même logique que votre script RLOO
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # Important pour la génération (PPO)

def build_dataset():
    ds = load_dataset("imdb", split="train[:1000]") # Même subset
    # On prépare juste les textes bruts, on tokenisera dans la boucle
    return [ " ".join(text.split()[:5]) for text in ds["text"] ]

train_prompts = build_dataset()

# --- 3. CHARGEMENT DES MODÈLES ---
print("--- Chargement des modèles ---")

# A. ACTEUR (Policy) - Initialisé avec SFT + Nouveau LoRA (Comme RLOO)
print("1. Chargement Acteur (SFT + LoRA)...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
# Fusion du SFT existant
model_sft = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH).merge_and_unload()

# Ajout d'une nouvelle couche LoRA fraîche (Comme 'rl_peft_config' dans RLOO)
rl_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)
actor = get_peft_model(model_sft, rl_peft_config)
actor.print_trainable_parameters()

# B. RÉFÉRENCE (Frozen) - C'est le modèle SFT fusionné
print("2. Chargement Référence...")
ref_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
ref_model = PeftModel.from_pretrained(ref_model, SFT_ADAPTER_PATH).merge_and_unload()
ref_model.eval()

# C. CRITIQUE (Value Model) - Spécifique PPO
# On l'initialise depuis le modèle SFT pour qu'il converge vite
print("3. Chargement Critique...")
critic = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=1).to(DEVICE)
critic.config.pad_token_id = tokenizer.eos_token_id
# On peut lui mettre un LoRA aussi pour économiser la mémoire, ou full finetune la tête
critic_peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, target_modules=["query_key_value"])
critic = get_peft_model(critic, critic_peft_config)

# D. REWARD MODEL (Juge Externe) - Identique RLOO
print("4. Chargement Reward Model (DistilBERT)...")
rm_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME, num_labels=2).to(DEVICE)
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
rm_model.eval()

# --- 4. OPTIMISEURS ---
optimizer = AdamW(list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(MAX_STEPS * WARMUP_RATIO), num_training_steps=MAX_STEPS)

# --- 5. FONCTIONS UTILITAIRES ---

def get_positive_score(texts):
    """Calcule le score POSITIVE (Index 1) via DistilBERT. Identique RLOO."""
    inputs = rm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = rm_model(**inputs)
    return outputs.logits[:, 1] # Score brut (logits)

def get_log_probs(model, input_ids, attention_mask):
    """Récupère les log-probs des tokens générés uniquement."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    # On décale input_ids de 1 vers la gauche pour aligner prediction et target
    labels = input_ids[:, 1:]
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

# --- 6. BOUCLE D'ENTRAÎNEMENT PPO ---
print(f"--- Démarrage PPO ({MAX_STEPS} steps) ---")
logs = {"reward": [], "actor_loss": [], "critic_loss": [], "kl": []}
global_step = 0

# On boucle indéfiniment sur le dataset mélangé jusqu'à atteindre MAX_STEPS
import itertools
data_loader = torch.utils.data.DataLoader(train_prompts, batch_size=BATCH_SIZE, shuffle=True)
data_iter = itertools.cycle(data_loader)

progress_bar = tqdm(total=MAX_STEPS)

# ... (Le début du script reste identique jusqu'à 'while global_step < MAX_STEPS:') ...

while global_step < MAX_STEPS:
    
    # --- A. ACCUMULATION DE GRADIENT ---
    batch_rewards = []
    batch_kls = []
    
    optimizer.zero_grad()
    
    for _ in range(GRAD_ACCUMULATION):
        
        # 1. Récupérer un batch
        prompts = next(data_iter)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        # 2. ROLLOUT
        with torch.no_grad():
            gen_output = actor.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        prompt_len = inputs["input_ids"].shape[1]
        response_seq = gen_output 
        attention_mask = (gen_output != tokenizer.pad_token_id).long()
        
        full_texts = tokenizer.batch_decode(gen_output, skip_special_tokens=True)
        
        # 3. CALCUL DES SIGNAUX (CORRIGÉ)
        with torch.no_grad():
            # a. Reward Externe
            rewards = get_positive_score(full_texts)
            
            # b. Valeur estimée (CORRECTION ICI)
            # Le Critique sort un score unique (batch_size,), pas une séquence.
            # On utilise ce score unique comme estimation pour toute la séquence.
            value_est = critic(input_ids=response_seq, attention_mask=attention_mask).logits.squeeze(-1)

            # c. KL Divergence
            ref_log_probs_all = get_log_probs(ref_model, response_seq, attention_mask)
            actor_log_probs_all = get_log_probs(actor, response_seq, attention_mask)
            
            start = prompt_len - 1
            ref_lp = ref_log_probs_all[:, start:]
            old_lp = actor_log_probs_all[:, start:]
            
            # Somme du KL sur la séquence (Sequence-level KL)
            kl_div = (old_lp - ref_lp).sum(dim=1)
            
            # REWARD TOTAL
            total_reward = rewards - (KL_BETA * kl_div)
            
            # AVANTAGE (Simple baseline subtraction)
            # R - V
            advantage = total_reward - value_est
        
        batch_rewards.append(rewards.mean().item())
        batch_kls.append(kl_div.mean().item())

        # 4. OPTIMISATION (CORRIGÉE)
        
        # Recalcul avec gradients
        curr_log_probs_all = get_log_probs(actor, response_seq, attention_mask)
        curr_lp = curr_log_probs_all[:, start:]
        
        # Ratio
        # On somme les log_probs pour avoir la probabilité de TOUTE la séquence
        # exp(sum(log_p_new) - sum(log_p_old)) = P_new(Seq) / P_old(Seq)
        log_prob_new = curr_lp.sum(dim=1)
        log_prob_old = old_lp.sum(dim=1)
        ratio = torch.exp(log_prob_new - log_prob_old)
        
        # PPO Clip Loss (Sequence Level)
        # On applique l'avantage unique à toute la séquence
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss (Le critique essaie de prédire le reward total)
        curr_value_est = critic(input_ids=response_seq, attention_mask=attention_mask).logits.squeeze(-1)
        critic_loss = F.mse_loss(curr_value_est, total_reward)
        
        loss = actor_loss + (VALUE_LOSS_COEF * critic_loss)
        
        loss = loss / GRAD_ACCUMULATION
        loss.backward()
    
    # --- B. UPDATE ---
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    global_step += 1
    progress_bar.update(1)
    
    avg_rew = sum(batch_rewards)/len(batch_rewards)
    avg_kl = sum(batch_kls)/len(batch_kls)
    logs["reward"].append(avg_rew)
    logs["kl"].append(avg_kl)
    logs["actor_loss"].append(actor_loss.item())
    logs["critic_loss"].append(critic_loss.item())
    
    if global_step % 10 == 0:
        progress_bar.set_description(f"R:{avg_rew:.2f} | KL:{avg_kl:.2f}")



# --- 7. SAUVEGARDE & PLOTS ---
print("Sauvegarde...")
actor.save_pretrained(f"./ppo_manual_comparison/final")
tokenizer.save_pretrained(f"./ppo_manual_comparison/final")

# Affichage comparatif
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(logs["reward"], label="PPO Reward")
plt.title("Evolution du Reward (DistilBERT Score)")
plt.xlabel("Steps")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(logs["kl"], label="KL Divergence", color="orange")
plt.title("Divergence vs SFT (Constraint)")
plt.xlabel("Steps")
plt.legend()

plt.show()