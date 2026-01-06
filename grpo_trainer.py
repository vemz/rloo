import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig  

# 1. Device and Environment Setup
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "EleutherAI/pythia-410m-deduped"
sft_adapter_path = "vemz/pythia-410m-sft-imdb"
output_dir = "./grpo_pythia_410m_explicit_pos"  

# 2. Load the Policy Model (The Actor)
# GRPO does not require a separate Critic/Value model, saving memory.
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map=None # Explicitly control device placement if needed
).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Generation models typically require left padding

# 3. Apply the SFT Adapter
# We merge the SFT adapter into the base model to create the starting policy.
model = PeftModel.from_pretrained(model, sft_adapter_path)
model = model.merge_and_unload()

# 4. Initialize the Reward Model (External Judge)
# This model will score the generations but is not trained itself.
rm_name = "lvwerra/distilbert-imdb"
reward_model = AutoModelForSequenceClassification.from_pretrained(rm_name, num_labels=2).to(device)
reward_tokenizer = AutoTokenizer.from_pretrained(rm_name)

# 5. Reward Function
# We encapsulate the reward model inference within this function.
def get_positive_score(prompts, completions, **kwargs):
    # RLOO/PPO might handle this internally, but for GRPO flexible rewards, we define it explicitly.
    # We construct the full sequences as the reward model expects (Prompt + Completion).
    inputs = [p + c for p, c in zip(prompts, completions)]
    
    # Tokenize the inputs for the reward model (DistilBERT)
    # We must ensure truncation to avoid errors if the generated text is too long.
    tokens = reward_tokenizer(
        inputs, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = reward_model(**tokens)
    
    # Extract the logits for the "Positive" class (index 1)
    # Using softmax to get a probability score  is standard for this RM.
    probs = torch.softmax(outputs.logits, dim=-1)
    scores = probs[:, 1].tolist() # Convert to list of floats
    return scores

# 6. Dataset Preparation
# GRPOTrainer requires a column named "prompt". 
def build_dataset(tokenizer):
    ds = load_dataset("imdb", split="train[:1000]") 
    def format_prompts(examples):
        # We take the first 5 words as the prompt, encouraging the model to complete the review.
        return {"prompt": [" ".join(text.split()[:5]) for text in examples["text"]]}
    ds = ds.map(format_prompts, batched=True, remove_columns=ds.column_names)
    return ds

train_dataset = build_dataset(tokenizer)

# 7. Define LoRA Config for Training
# This config will be applied to the Policy Model during the RL phase.
rl_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)

# 8. Configure Training Arguments (GRPOConfig)
# Key parameter changes: num_generations, beta
grpo_config = GRPOConfig(
    output_dir=output_dir,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_generations=4,          # MANDATORY: The group size (G) for relative scoring.
    max_prompt_length=128,      # Limits the input prompt size
    max_completion_length=32,   # Limits the generated text size
    beta=0.02,                  # KL Divergence penalty coefficient
    per_device_train_batch_size=2, # Actual batch size = 2 * num_generations (4) = 8 sequences per step
    gradient_accumulation_steps=16,
    max_steps=200,
    logging_steps=10,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1          
)

# 9. Initialize the Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=get_positive_score, 
    args=grpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer, 
    peft_config=rl_peft_config, 
)

# 10. Execute Training
trainer.train()

# 11. Save Artifacts
trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")