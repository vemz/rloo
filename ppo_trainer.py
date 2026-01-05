import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import random

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
LR = 1e-5
EPOCHS = 1
BATCH_SIZE = 4
PPO_EPOCHS = 4  # Nombre de fois qu'on optimise sur le même batch généré
CLIP_EPS = 0.2  # Le "Clip" de PPO (0.8 - 1.2)
KL_BETA = 0.02  # Pénalité pour ne pas trop s'éloigner du modèle de base

# --- 2. CHARGEMENT DES MODÈLES ---
print("Chargement des modèles...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# A. ACTEUR (Policy) - Celui qu'on entraîne
actor = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
actor_opt = AdamW(actor.parameters(), lr=LR)

# B. CRITIQUE (Value Model) - Estime la qualité de l'état
critic = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
critic.config.pad_token_id = tokenizer.eos_token_id
critic_opt = AdamW(critic.parameters(), lr=LR)

# C. REFERENCE (Pour le KL - Gelé)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
ref_model.eval()

# D. REWARD MODEL (Le Juge - Gelé)
reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
reward_model.eval()

# --- 3. FONCTIONS UTILITAIRES PPO ---

def get_log_probs(model, input_ids, attention_mask):
    """Calcule les log-probabilités des tokens dans une séquence."""
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :] # On enlève le dernier logit car il prédit le futur
    input_ids = input_ids[:, 1:]      # On décale les inputs d'un cran
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    # On va chercher la proba exacte du token qui a été choisi
    selected_log_probs = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
    return selected_log_probs

def compute_rewards(texts):
    """Appelle le Reward Model (Juge)"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # On suppose que le reward model sort un score scalaire direct
        rewards = outputs.logits.squeeze(-1)
    return rewards

# --- 4. DATASET ---
ds = load_dataset("imdb", split="train").select(range(100)) # Petit test

# --- 5. BOUCLE D'ENTRAÎNEMENT PRINCIPALE ---
print("Démarrage de l'entraînement PPO Manuel...")

for step, sample in enumerate(tqdm(ds)):
    # -------------------------------------------------------
    # PHASE 1 : ROLLOUT (Génération de données)
    # -------------------------------------------------------
    
    # Préparation du prompt
    prompt_text = " ".join(sample["text"].split()[:5]) # 5 premiers mots
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    # Génération de la réponse par l'Acteur
    with torch.no_grad():
        gen_tokens = actor.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    # On récupère le texte complet
    full_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    response_tokens = gen_tokens # Tout la séquence
    attention_mask = (response_tokens != tokenizer.pad_token_id).long()

    # -------------------------------------------------------
    # PHASE 2 : ÉVALUATION (Calcul des signaux)
    # -------------------------------------------------------
    
    with torch.no_grad():
        # A. Calcul du Reward (Qualité externe)
        reward_score = compute_rewards([full_text])[0]
        
        # B. Calcul de la Value (Estimation interne du Critique)
        value_est = critic(input_ids=response_tokens, attention_mask=attention_mask).logits.squeeze()
        
        # C. Calcul des Log Probs de référence (pour le KL)
        ref_log_probs = get_log_probs(ref_model, response_tokens, attention_mask)
        
        # D. Calcul des Log Probs "Old" (celles qui ont généré le texte)
        old_log_probs = get_log_probs(actor, response_tokens, attention_mask)

    # E. Calcul de l'Avantage (Simplifié : Reward - Value)
    # Dans une vraie implém complète, on utiliserait GAE, mais R - V suffit pour démarrer
    advantage = reward_score - value_est.item()
    
    # Calcul du KL (pénalité)
    kl_div = (old_log_probs - ref_log_probs).mean()
    
    # Reward total (Reward score - pénalité KL)
    total_reward = reward_score - (KL_BETA * kl_div.item())

    # -------------------------------------------------------
    # PHASE 3 : OPTIMISATION PPO (Plusieurs passes)
    # -------------------------------------------------------
    
    for _ in range(PPO_EPOCHS):
        # 1. On recalcule les probas actuelles (car le modèle change à chaque petit tour)
        current_log_probs = get_log_probs(actor, response_tokens, attention_mask)
        
        # 2. Ratio (Importance Sampling) : exp(new - old)
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # 3. PPO Clipped Loss (La fameuse formule)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 4. Value Loss (Le critique doit apprendre à prédire le reward)
        # On veut que critic(x) soit proche du reward réel
        curr_value = critic(input_ids=response_tokens, attention_mask=attention_mask).logits.squeeze()
        critic_loss = F.mse_loss(curr_value, torch.tensor(total_reward, device=DEVICE))
        
        # 5. Backprop
        loss = actor_loss + 0.5 * critic_loss
        
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        loss.backward(retain_graph=True) # retain_graph car on réutilise les tenseurs
        actor_opt.step()
        critic_opt.step()

    # Logs simple
    if step % 10 == 0:
        print(f"Step {step}: Reward={reward_score:.3f}, Loss={loss.item():.3f}, KL={kl_div.item():.3f}")

# Sauvegarde manuelle
actor.save_pretrained("my_ppo_manual_model")
print("Terminé !")