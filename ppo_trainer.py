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
import re 

# =============================================================================
# 1. CONFIGURATION 
# ========================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

# Modèles
BASE_MODEL_NAME = "EleutherAI/pythia-410m-deduped"
SFT_ADAPTER_PATH = "vemz/pythia-410m-sft-imdb" 
REWARD_MODEL_NAME = "lvwerra/distilbert-imdb"

# Hyperparamètres
LEARNING_RATE = 3e-5
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1
KL_BETA = 0.05            
BATCH_SIZE = 2             
GRAD_ACCUMULATION = 16     
MAX_STEPS = 100            
MAX_NEW_TOKENS = 48      

# Params PPO
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5

# ==============================================================================
# 2. INITIALISATION WANDB
# =============================================================================
wandb.login()
run = wandb.init(
    project="ppo-manual-comparison",
    name="ppo-strict-match-rloo-v3",
    config={
        "algo": "PPO_Manual",
        "reward_fn": "Combined (Pos + Rep + Len)",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "kl_beta": KL_BETA,
        "max_steps": MAX_STEPS,
        "model": "pythia-410m"
    }
)

# ==============================================================================
# 3. DATASET & CLEANING 
# ==============================================================================
print("Chargement dataset...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# Fonction de nettoyage 
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_dataset():
    ds = load_dataset("imdb", split="train[:300]")  # Taille réduite à 300 
    # Prompt allongé à 10 mots + Cleaning
    return [ " ".join(clean_text(text).split()[:10]) for text in ds["text"] ]

train_prompts = build_dataset()
data_loader = torch.utils.data.DataLoader(train_prompts, batch_size=BATCH_SIZE, shuffle=True)
data_iter = itertools.cycle(data_loader)

# ==============================================================================
# 4. MODÈLES
# ==============================================================================
print("Architecture PPO ")

# POLICY
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
try:
    model_sft = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH).merge_and_unload()
    print("-> SFT chargé.")
except:
    model_sft = base_model

rl_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)
actor = get_peft_model(model_sft, rl_peft_config)

# REFERENCE
ref_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
try:
    ref_model = PeftModel.from_pretrained(ref_model, SFT_ADAPTER_PATH).merge_and_unload()
except:
    pass
ref_model.eval()

# CRITIC
critic = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=1).to(DEVICE)
critic.config.pad_token_id = tokenizer.eos_token_id
critic_peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, target_modules=["query_key_value"])
critic = get_peft_model(critic, critic_peft_config)

# REWARD MODEL
rm_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME, num_labels=2).to(DEVICE)
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
rm_model.eval()

optimizer = AdamW(list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(MAX_STEPS * WARMUP_RATIO), num_training_steps=MAX_STEPS)

# ==============================================================================
# 5. NOUVELLES FONCTIONS DE RECOMPENSE
# ===================================================================

def get_positive_score(texts):
    """Score DistilBERT"""
    inputs = rm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = rm_model(**inputs)
    return outputs.logits[:, 1]

# Pénalité de repetitions
def repetition_penalty_reward(completions_text):
    
    values = []
    for c in completions_text:
        words = c.lower().split()
        if len(words) == 0:
            values.append(0.0)
        else:
            values.append(len(set(words)) / len(words))
    return torch.tensor(values, device=DEVICE)

def compute_combined_reward(full_texts, completions_text):
    
    # 1. Score Positif (Sur le texte entier Prompt + Completion)
    pos_score = get_positive_score(full_texts)
    
    # 2. Score Répétition (Sur la completion uniquement)
    rep_score = repetition_penalty_reward(completions_text)
    
    # 3. Pénalité Longueur (Sur la completion uniquement)
    lengths = torch.tensor([len(c.split()) for c in completions_text], device=DEVICE)
    short_penalty = (lengths < 5).float() * -2.0
    
    total_reward = pos_score + (0.5 * rep_score) + short_penalty
    return total_reward

def get_log_probs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

# ==============================================================================
# 6. BOUCLE PPO
# ==============================================================================
print(f"\n--- Démarrage PPO (Combined Reward) ---")
data_loader = torch.utils.data.DataLoader(train_prompts, batch_size=BATCH_SIZE, shuffle=True)
data_iter = itertools.cycle(data_loader)
progress_bar = tqdm(total=MAX_STEPS)

global_step = 0

while global_step < MAX_STEPS:
    
    batch_rewards = []
    batch_kls = []
    optimizer.zero_grad()
    
    for _ in range(GRAD_ACCUMULATION):
        
        prompts = next(data_iter)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        
        with torch.no_grad():
            gen_output = actor.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, # 48
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        prompt_len = inputs["input_ids"].shape[1]
        response_seq = gen_output
        attention_mask = (gen_output != tokenizer.pad_token_id).long()
        
        # Décodage : On a besoin du texte entier et  de la completion
        full_texts = tokenizer.batch_decode(gen_output, skip_special_tokens=True)
        # Pour extraire la completion, on décode la partie générée
        completions_ids = gen_output[:, prompt_len:]
        completions_text = tokenizer.batch_decode(completions_ids, skip_special_tokens=True)
        
        # Calcul des Signaux
        with torch.no_grad():
            
            rewards = compute_combined_reward(full_texts, completions_text)
        
            
            value_est = critic(input_ids=response_seq, attention_mask=attention_mask).logits.squeeze(-1)

           
            actor.eval()
            ref_log_probs_all = get_log_probs(ref_model, response_seq, attention_mask)
            actor_log_probs_all = get_log_probs(actor, response_seq, attention_mask)
            actor.train()
            
            start = prompt_len - 1
            ref_lp = ref_log_probs_all[:, start:]
            old_lp = actor_log_probs_all[:, start:]
            # 1. On récupère le masque de la partie générée uniquement
            # attention_mask vaut 1 pour les mots réels, 0 pour le padding
            gen_mask = attention_mask[:, start:]
            
            # 2. On calcule le KL token par token
            kl_tokens = (old_lp - ref_lp)
            
            # 3. On annule le KL sur les tokens de padding (multiplication par 0)
            kl_tokens = kl_tokens * gen_mask
            
            # 4. On fait la somme
            kl_div = kl_tokens.sum(dim=1)
            
            
            total_reward = rewards - (KL_BETA * kl_div)
            advantage = total_reward - value_est
        
        batch_rewards.append(rewards.mean().item())
        batch_kls.append(kl_div.mean().item())

        # Optimisation
        curr_log_probs_all = get_log_probs(actor, response_seq, attention_mask)
        curr_lp = curr_log_probs_all[:, start:]
        
        log_prob_new = curr_lp.sum(dim=1)
        log_prob_old = old_lp.sum(dim=1)
        ratio = torch.exp(log_prob_new - log_prob_old)
        
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        curr_val_est = critic(input_ids=response_seq, attention_mask=attention_mask).logits.squeeze(-1)
        critic_loss = F.mse_loss(curr_val_est, total_reward)
        
        loss = actor_loss + (VALUE_LOSS_COEF * critic_loss)
        loss = loss / GRAD_ACCUMULATION
        loss.backward()
    
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    avg_rew = sum(batch_rewards)/len(batch_rewards)
    std_rew = np.std(batch_rewards)
    avg_kl = sum(batch_kls)/len(batch_kls)
    
    wandb.log({
        "train/reward": avg_rew,
        "train/reward_std" : std_rew,
        "train/kl": avg_kl,
        "train/loss": loss.item() * GRAD_ACCUMULATION,
        "global_step": global_step
    })
    
    global_step += 1
    progress_bar.update(1)
    if global_step % 10 == 0:
        progress_bar.set_description(f"R:{avg_rew:.2f} | KL:{avg_kl:.2f}")

print("\nSauvegarde...")
output_dir = "./ppo_manual_reward_penalized"
actor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
wandb.finish()