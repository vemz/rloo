import torch
import types
import re
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl.models.utils import disable_gradient_checkpointing
from peft import PeftModel, LoraConfig, TaskType
from trl import RLOOTrainer, RLOOConfig
from accelerate.utils import gather

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "EleutherAI/pythia-410m-deduped"
sft_adapter_path = "vemz/pythia-410m-sft-imdb"
output_dir = "./rloo_pythia_weighted_baseline"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
).to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = PeftModel.from_pretrained(model, sft_adapter_path)
model = model.merge_and_unload()

rm_name = "lvwerra/distilbert-imdb"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_name,
    num_labels=2
).to(device)
reward_tokenizer = AutoTokenizer.from_pretrained(rm_name)

def get_positive_score(prompts, completions, **kwargs):
    inputs = [p + c for p, c in zip(prompts, completions)]
    tokens = reward_tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = reward_model(**tokens)
    return outputs.logits[:, 1]

def repetition_penalty_reward(completions):
    values = []
    for c in completions:
        words = c.lower().split()
        values.append(len(set(words)) / max(1, len(words)))
    return torch.tensor(values, device=device)

def combined_reward(prompts, completions, **kwargs):
    pos = get_positive_score(prompts, completions)
    rep = repetition_penalty_reward(completions)
    lengths = torch.tensor([len(c.split()) for c in completions], device=device)
    short_penalty = (lengths < 5).float() * -2.0
    return pos + 0.5 * rep + short_penalty

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_dataset():
    ds = load_dataset("imdb", split="train[:300]")
    
    def format_prompts(examples):
        return {
            "prompt": [
                " ".join(clean_text(text).split()[:10])
                for text in examples["text"]
            ]
        }
    return ds.map(format_prompts, batched=True, remove_columns=ds.column_names)

train_dataset = build_dataset()

rl_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
)

rloo_config = RLOOConfig(
    output_dir=output_dir,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_generations=4,
    beta=0.05,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    max_steps=100,
    logging_steps=10,
    bf16=False,
    fp16=False,
    gradient_checkpointing=True,
    max_completion_length=48,
    save_strategy="steps",
    save_steps=40,
    save_total_limit=1,
)

def pad(tensors, padding_value, padding_side):
    if padding_side == "left":
        return pad_sequence([t.flip(0) for t in tensors], batch_first=True, padding_value=padding_value).flip(1)
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

def probability_weighted_generation_step(self, inputs):
    device = self.accelerator.device
    mode = "train" if self.model.training else "eval"

    prompts = [x["prompt"] for x in inputs]
    
    gen_output = self._generate(prompts)
    
    if len(gen_output) == 3:
        prompt_ids_list, completion_ids_list, completions = gen_output
    else:
        prompt_ids_list, completion_ids_list = gen_output
        completions = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)

    prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
    prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
    prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
    prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
    
    completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
    completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
    completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
    completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

    if self.mask_truncated_completions:
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

    with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
        old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
        )
        old_logps = (old_per_token_logps * completion_mask).sum(1) 

        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )
        else:
            ref_per_token_logps = None

    rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
    rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

    if self.reward_clip_range:
        rewards = rewards.clamp(min=self.reward_clip_range[0], max=self.reward_clip_range[1])

    if self.beta != 0.0:
        per_token_kl = old_per_token_logps - ref_per_token_logps
        kl = (per_token_kl * completion_mask).sum(-1)
        kl = gather(kl)
        rewards = rewards - self.beta * kl
        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

    num_generations = self.num_generations if mode == "train" else self.num_generations_eval
    
    if num_generations > 1:
        grouped_logps = old_logps.view(-1, num_generations)
        grouped_rewards = rewards.view(-1, num_generations)
        
        weights = torch.nn.functional.softmax(grouped_logps, dim=1)
        weighted_baseline = (weights * grouped_rewards).sum(dim=1, keepdim=True)
        
        advantages = rewards - weighted_baseline.expand(-1, num_generations).reshape(-1)
    else:
        advantages = torch.zeros_like(rewards)

    grouped_rewards = rewards.view(-1, num_generations)
    mean_grouped_rewards = grouped_rewards.mean(dim=1)
    std_rewards = grouped_rewards.std(dim=1) if num_generations > 1 else torch.zeros_like(mean_grouped_rewards)

    self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
    self._metrics[mode]["reward_std"].append(std_rewards.mean().item())

    for i, reward_func_name in enumerate(self.reward_func_names):
        mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        
        std_rewards_func = rewards_per_func[:, i].std().item()
        self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards_func)

    if self.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

    process_slice = slice(
        self.accelerator.process_index * len(prompts),
        (self.accelerator.process_index + 1) * len(prompts),
    )
    advantages = advantages[process_slice]

    output = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "old_logps": old_logps,
        "advantages": advantages,
    }
    return output

trainer = RLOOTrainer(
    model=model,
    reward_funcs=combined_reward,
    args=rloo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=rl_peft_config,
)

trainer._generate_and_score_completions = types.MethodType(probability_weighted_generation_step, trainer)

trainer.train()
trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")