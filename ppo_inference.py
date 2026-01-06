import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "EleutherAI/pythia-410m-deduped"

ADAPTER_PATH = "./ppo_manual_comparison/final" 
REWARD_MODEL_NAME = "lvwerra/distilbert-imdb"

# Prompts de test : Certains neutres, certains négatifs pour voir si le modèle force le positif
TEST_PROMPTS = [
    "The movie was",
    "I really hated this film because",
    "Honestly, the acting was",
    "I went to the cinema and",
    "This is the worst",
    "The director tried to"
]

# --- 2. CHARGEMENT DU JUGE (Pour le score) ---
print("Chargement du Juge (DistilBERT)...")
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
rm_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME).to(DEVICE)

def get_sentiment_score(texts):
    """Retourne le score de positivité (0 à 100%)"""
    inputs = rm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = rm_model(**inputs)
    # On prend le softmax de la classe "POSITIVE" (index 1)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1].cpu().numpy() * 100

# --- 3. CHARGEMENT DU MODÈLE DE GÉNÉRATION ---
print("Chargement du modèle de base...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# On charge d'abord le modèle de base (SFT ou Base)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)

# --- 4. GÉNÉRATION "AVANT" (Base Model) ---
print("Génération avec le modèle DE BASE...")
base_results = []

for prompt in TEST_PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    base_results.append(text)

# --- 5. APPLICATION DE L'ADAPTATEUR PPO ---
print(f"Chargement de l'adaptateur depuis {ADAPTER_PATH}...")
# On applique le LoRA par dessus le modèle de base
ppo_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
ppo_model.eval()

# --- 6. GÉNÉRATION "APRÈS" (PPO Model) ---
print("Génération avec le modèle PPO/RLOO...")
ppo_results = []

for prompt in TEST_PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = ppo_model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=True, 
            temperature=0.7, # Même température pour comparer
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ppo_results.append(text)

# --- 7. COMPARAISON ET AFFICHAGE ---
print("\n--- RÉSULTATS COMPARATIFS ---")

base_scores = get_sentiment_score(base_results)
ppo_scores = get_sentiment_score(ppo_results)

data = []
for i, prompt in enumerate(TEST_PROMPTS):
    data.append({
        "Prompt": prompt,
        "Base Model Output": base_results[i].replace(prompt, " [..] "),
        "Base Score": f"{base_scores[i]:.1f}%",
        "PPO Model Output": ppo_results[i].replace(prompt, " [..] "),
        "PPO Score": f"{ppo_scores[i]:.1f}%",
        "Gain": f"{ppo_scores[i] - base_scores[i]:.1f}%"
    })

df = pd.DataFrame(data)

from IPython.display import display
display(df)


for row in data:
    print("-" * 80)
    print(f"PROMPT: {row['Prompt']}")
    print(f"BASE ({row['Base Score']}): ...{row['Base Model Output']}")
    print(f"PPO  ({row['PPO Score']}): ...{row['PPO Model Output']}")