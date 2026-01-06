import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import itertools
import wandb  

# ==============================================================================
# 1. CONFIGURATION 
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

# Noms des modèles
BASE_MODEL_NAME = "EleutherAI/pythia-410m-deduped"
SFT_ADAPTER_PATH = "vemz/pythia-410m-sft-imdb" 
REWARD_MODEL_NAME = "lvwerra/distilbert-imdb"

# Hyperparamètres
LEARNING_RATE = 3e-5
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1
KL_BETA = 0.05             # Pénalité KL 
BATCH_SIZE = 2             # Petit batch pour GPU modeste
GRAD_ACCUMULATION = 16     # Accumulation pour simuler un grand batch (Total = 32)
MAX_STEPS = 100            # Durée de l'entraînement
MAX_NEW_TOKENS = 32        # Longueur générée

# Params PPO
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5

# ==============================================================================
# 2. INITIALISATION WANDB
# ==============================================================================
wandb.login() 
run = wandb.init(
    project="ppo-manual-pytorch",
    name="pythia-410m-ppo-vs-rloo",
    config={
        "algo": "PPO_Manual",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "kl_beta": KL_BETA,
        "max_steps": MAX_STEPS,
        "model": "pythia-410m"
    }
)

# ==============================================================================
# 3. CHARGEMENT DU DATASET
# ==============================================================================
print("Chargement dataset...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # PPO génère à droite, donc padding à gauche

def build_dataset():
    ds = load_dataset("imdb", split="train[:1000]") 
    # On garde juste le début des phrases comme "Prompt"
    return [ " ".join(text.split()[:5]) for text in ds["text"] ]

train_prompts = build_dataset()
data_loader = torch.utils.data.DataLoader(train_prompts, batch_size=BATCH_SIZE, shuffle=True)
data_iter = itertools.cycle(data_loader) # Cycle infini pour éviter StopIteration

# ==============================================================================
# 4. CHARGEMENT DES MODÈLES
# ==============================================================================
print("--- Construction de l'architecture PPO ---")

# A. ACTEUR (Policy)
# 1. On charge la base
print("1. Acteur (Base + SFT + LoRA)...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)

# 2. On charge l'adaptateur SFT existant 
try:
    model_sft = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model_sft = model_sft.merge_and_unload() # FUSION CRITIQUE
    print("   -> SFT Adapter chargé et fusionné.")
except Exception as e:   # si n'existe pas on prend celui de base
    print(f"   -> Attention : SFT introuvable ({e}), on part du modèle de base.")
    model_sft = base_model

# 3. nouvelle couche LoRA  pour le PPO
rl_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)
actor = get_peft_model(model_sft, rl_peft_config)
actor.print_trainable_parameters()

# B. REFERENCE
#identique à l'acteur AVANT entraînement PPO 
print("2. Référence (Base + SFT)...")
ref_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
try:
    ref_model = PeftModel.from_pretrained(ref_model, SFT_ADAPTER_PATH).merge_and_unload()
except:
    pass
ref_model.eval()

# C. CRITIQUE (Value Model)
print("3. Critique (Value Model)...")
critic = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=1).to(DEVICE)
critic.config.pad_token_id = tokenizer.eos_token_id
# Petit LoRA sur le critique pour économiser la VRAM
critic_peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, target_modules=["query_key_value"])
critic = get_peft_model(critic, critic_peft_config)

# D. REWARD MODEL (Juge)
print("4. Juge (DistilBERT)...")
rm_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME, num_labels=2).to(DEVICE)
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
rm_model.eval()

# ==============================================================================
# 5. UTILITAIRES & OPTIMISEURS
# ==============================================================================
optimizer = AdamW(list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(MAX_STEPS * WARMUP_RATIO), num_training_steps=MAX_STEPS)

def get_positive_score(texts):
    """Récupère le score POSITIF (index 1) du Juge DistilBERT"""
    inputs = rm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = rm_model(**inputs)
    return outputs.logits[:, 1] # Logits bruts

def get_log_probs(model, input_ids, attention_mask):
    """Log-probs des tokens"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

# ==============================================================================
# 6. BOUCLE D'ENTRAÎNEMENT 
# ==============================================================
print(f"\n--- Démarrage PPO ({MAX_STEPS} steps) ---")
progress_bar = tqdm(total=MAX_STEPS)
global_step = 0

while global_step < MAX_STEPS:
    
    batch_rewards = []
    batch_kls = []
    
    optimizer.zero_grad()
    
    # Accumulation de Gradient 
    for _ in range(GRAD_ACCUMULATION):
        
        # 1. ROLLOUT
        prompts = next(data_iter)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
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
        
        # 2. rewards and value calcul
        with torch.no_grad():
            # Reward 
            rewards = get_positive_score(full_texts)
            
            # b. Value (Critique) -> Score unique par phrase 
            value_est = critic(input_ids=response_seq, attention_mask=attention_mask).logits.squeeze(-1)
            
            # c. KL Divergence 
            ref_log_probs_all = get_log_probs(ref_model, response_seq, attention_mask)
            actor_log_probs_all = get_log_probs(actor, response_seq, attention_mask)
            
            start = prompt_len - 1
            ref_lp = ref_log_probs_all[:, start:]
            old_lp = actor_log_probs_all[:, start:]
            
            # KL par séquence (somme sur les tokens)
            kl_div = (old_lp - ref_lp).sum(dim=1)
            
            # d. Avantage
            total_reward = rewards - (KL_BETA * kl_div)
            advantage = total_reward - value_est
        
        # xtats pour logging
        batch_rewards.append(rewards.mean().item())
        batch_kls.append(kl_div.mean().item())

        # 3. calcul de la loss
        # probs avec gradients activés
        curr_log_probs_all = get_log_probs(actor, response_seq, attention_mask)
        curr_lp = curr_log_probs_all[:, start:]

        log_prob_new = curr_lp.sum(dim=1)
        log_prob_old = old_lp.sum(dim=1)
        ratio = torch.exp(log_prob_new - log_prob_old)
        # PPO Loss
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic Loss
        curr_val_est = critic(input_ids=response_seq, attention_mask=attention_mask).logits.squeeze(-1)
        critic_loss = F.mse_loss(curr_val_est, total_reward)
        
        loss = actor_loss + (VALUE_LOSS_COEF * critic_loss)
        loss = loss / GRAD_ACCUMULATION
        loss.backward()

    # --- UPDATE STEP ---
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    # --- WANDB LOGGING ---
    avg_rew = sum(batch_rewards)/len(batch_rewards)
    avg_kl = sum(batch_kls)/len(batch_kls)
    
    wandb.log({
        "train/reward": avg_rew,
        "train/kl": avg_kl,
        "train/loss": loss.item() * GRAD_ACCUMULATION, 
        "train/actor_loss": actor_loss.item(),
        "train/critic_loss": critic_loss.item(),
        "train/learning_rate": scheduler.get_last_lr()[0],
        "global_step": global_step
    })
    
    global_step += 1
    progress_bar.update(1)
    progress_bar.set_description(f"R:{avg_rew:.2f} | KL:{avg_kl:.2f}")

# ==============================================================================
# 7. FIN ET SAUVEGARDE
# ==============================================================================
print("\nSauvegarde...")
output_dir = "./ppo_manual_final"
actor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
wandb.finish()
print(f"Entraînement terminé ! Modèle sauvegardé dans {output_dir}")