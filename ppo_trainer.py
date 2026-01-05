import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt # Import pour les graphes
import numpy as np

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
LR = 1e-5
PPO_EPOCHS = 4
CLIP_EPS = 0.2
KL_BETA = 0.02
MAX_STEPS = 100 # On fait 100 pas pour avoir de quoi tracer

# Dictionnaire pour stocker les logs (Ce que vous voulez tracer)
logs = {
    "reward": [],
    "step_count": [],        # On utilisera la longueur de la réponse générée
    "eval reward (sum)": [], # Reward sur le jeu de test
    "eval step_count": []
}

# --- 2. CHARGEMENT ---
print("Chargement des modèles...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

actor = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
actor_opt = AdamW(actor.parameters(), lr=LR)

critic = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
critic.config.pad_token_id = tokenizer.eos_token_id
critic_opt = AdamW(critic.parameters(), lr=LR)

ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
ref_model.eval()

reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
reward_model.eval()

# --- 3. FONCTIONS ---
def get_log_probs(model, input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :]
    input_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)

def compute_rewards(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        return outputs.logits.squeeze(-1)

# Fonction d'évaluation (Pour remplir les logs "eval")
def evaluate(actor, val_ds):
    actor.eval()
    rewards = []
    lengths = []
    # On teste sur 4 exemples seulement pour aller vite
    for sample in val_ds.select(range(4)):
        inputs = tokenizer(sample["text"][:50], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = actor.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        r = compute_rewards([full_text])[0].item()
        rewards.append(r)
        lengths.append(len(gen[0])) # Longueur totale en tokens
    actor.train()
    return np.mean(rewards), np.mean(lengths)

# --- 4. DATASET ---
# Train set et Test set
ds_train = load_dataset("imdb", split="train").select(range(MAX_STEPS))
ds_test = load_dataset("imdb", split="test").select(range(10)) # Petit set de test

# --- 5. BOUCLE D'ENTRAÎNEMENT ---
print(f"Démarrage PPO sur {MAX_STEPS} steps...")

for step, sample in enumerate(tqdm(ds_train)):
    
    # --- A. ROLLOUT ---
    prompt_text = " ".join(sample["text"].split()[:5])
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        gen_tokens = actor.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    response_tokens = gen_tokens
    attention_mask = (response_tokens != tokenizer.pad_token_id).long()

    # --- B. LOGGING TRAIN (Ici on remplit vos listes) ---
    with torch.no_grad():
        reward_score = compute_rewards([full_text])[0]
    
    # On ajoute aux logs
    logs["reward"].append(reward_score.item())
    logs["step_count"].append(response_tokens.shape[1]) # Longueur de la séquence

    # --- C. UPDATE PPO ---
    with torch.no_grad():
        value_est = critic(input_ids=response_tokens, attention_mask=attention_mask).logits.squeeze()
        ref_log_probs = get_log_probs(ref_model, response_tokens, attention_mask)
        old_log_probs = get_log_probs(actor, response_tokens, attention_mask)

    advantage = reward_score - value_est.item()
    
    for _ in range(PPO_EPOCHS):
        current_log_probs = get_log_probs(actor, response_tokens, attention_mask)
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        curr_value = critic(input_ids=response_tokens, attention_mask=attention_mask).logits.squeeze()
        critic_loss = F.mse_loss(curr_value, torch.tensor(reward_score - (KL_BETA * 0), device=DEVICE))
        
        loss = actor_loss + 0.5 * critic_loss
        
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        loss.backward(retain_graph=True)
        actor_opt.step()
        critic_opt.step()

    # --- D. EVALUATION (Tous les 10 steps) ---
    if step % 10 == 0:
        eval_rew, eval_len = evaluate(actor, ds_test)
        logs["eval reward (sum)"].append(eval_rew)
        logs["eval step_count"].append(eval_len)

# --- 6. VOTRE CODE DE TRAÇAGE ---
print("Entraînement terminé. Affichage des graphes...")

plt.figure(figsize=(10, 10))

# 1. Training Reward
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("Training rewards (step by step)")
plt.xlabel("Step")
plt.ylabel("Reward Score")

# 2. Training Length (Step count)
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Generated Length (Training)")
plt.xlabel("Step")

# 3. Eval Reward
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"]) # Note: C'est une moyenne tous les 10 steps
plt.title("Eval Reward (Every 10 steps)")
plt.xlabel("Eval Epoch")

# 4. Eval Length
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Eval Length (Every 10 steps)")

plt.tight_layout()
plt.show()