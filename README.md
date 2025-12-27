# RLHF Fine-Tuning: PPO vs. RLOO (Back to Basics)


![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-TRL-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This project benchmarks and compares two RLHF algorithms for aligning language models on the IMDB sentiment task:

1. **PPO (Proximal Policy Optimization):** The standard RLHF method, using an Actor-Critic setup with a value function (Critic) to reduce variance.
2. **RLOO (REINFORCE Leave-One-Out):** A recent, simpler alternative that removes the Critic and uses a leave-one-out baseline computed from multiple generations per prompt.

The experiments are performed on the [EleutherAI/pythia-410m-deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped) model, fine-tuned to generate **positive movie reviews** using the IMDB dataset. The project demonstrates that RLOO can match or outperform PPO in this setting, while being more memory and compute-efficient.

Key features:
- Direct comparison of PPO and RLOO on the same task and model.
- Ablation on the number of generations $k$ for RLOO and on the baseline computation.
- All training and evaluation code is provided for reproducibility.

## Background: PPO vs. RLOO

**PPO** relies on a Critic (value function) to estimate expected rewards and stabilize policy gradients, but this adds memory and tuning complexity.

**RLOO** eliminates the Critic. For each prompt, $k$ completions are generated; the reward for each is compared to the mean reward of the other $k-1$ completions (leave-one-out baseline), reducing variance without a value network.

This project provides practical scripts and analysis to help understand the trade-offs between these two RLHF approaches.

## Installation

This project relies on the Hugging Face `trl` library and related tools.

```bash
# 1. Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# 2. Install dependencies
pip install torch transformers datasets accelerate numpy tqdm matplotlib seaborn
pip install trl==0.8.6 

---

## Usage and Experiments

### 1. Fine-tuning SFT (Supervised Fine-Tuning)
The script `sft.py` trains a Pythia-410M model on IMDB to generate positive reviews using LoRA. The SFT model serves as a starting point for PPO and RLOO.

### 2. RLHF with PPO and RLOO
- **PPO**: Uses a Critic (Value Function) to stabilize training. (Script not included here, but see the literature and the `trl` library)
- **RLOO**: Implemented in `rloo_trainer.py`, removes the Critic and uses a leave-one-out baseline over k generations.

#### Testing different k (RLOO)
In `rloo_trainer.py`, the `num_generations` (k) parameter controls the number of completions generated per prompt for the leave-one-out baseline. Several values of k were tested (e.g., k=2, 4) to observe the impact on training stability and performance.

#### Changing the baseline
The code allows comparing the leave-one-out baseline (mean of the other k-1 rewards) to other strategies (global mean, constant baseline, etc.) by modifying the advantage function in the trainer.

#### Comparison with PPO
Results obtained with RLOO (for different k and baselines) are compared to those of PPO (see TRL reference). Main metrics are the proportion of positive reviews generated and training stability.

### 3. Generation and Evaluation
- `sft_inference.py`: Allows comparison of outputs from the base and SFT models on the same prompts.
- `rloo_inference.py`: Allows comparison of outputs from the SGT and RLOO models on the same prompts.

---

## References
- [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)

---
