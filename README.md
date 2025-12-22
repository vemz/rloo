# RLHF Fine-Tuning: PPO vs. RLOO (Back to Basics)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-TRL-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📄 Project Overview

This research project aims to reproduce and compare two alignment algorithms for Large Language Models (LLMs) using **Reinforcement Learning from Human Feedback (RLHF)**:

1.  **PPO (Proximal Policy Optimization):** The current industry standard (used in ChatGPT, Llama-2-Chat), relying on a complex Actor-Critic architecture.
2.  **RLOO (REINFORCE Leave-One-Out):** A simpler, more memory-efficient method proposed in the paper *"Back to Basics: PPO against Reinforced Leave-One-Out"* (Ahmadian et al., 2024).

The primary objective is to fine-tune a **GPT-2** model to generate **positive movie reviews** on the **IMDB dataset**. The project demonstrates that RLOO can achieve performance comparable to or better than PPO without requiring a "Critic" (Value Function) network, thereby significantly reducing VRAM usage and computational complexity.

## 📚 Theoretical Background

### The PPO Bottleneck
PPO is an **Actor-Critic** method. It requires loading four models into memory during training: the Actor, the Critic, the Reference Model, and the Reward Model. The **Critic (Value Function)** attempts to predict future rewards token-by-token to reduce gradient variance. However, training the Critic is often unstable and computationally expensive.

### The RLOO Solution
RLOO removes the Critic entirely. It revisits the **REINFORCE** algorithm but stabilizes it using a statistical baseline derived from the generated data itself:
* **Sampling:** For every prompt, the model generates $k$ independent completions (e.g., $k=4$).
* **Leave-One-Out Baseline:** To calculate the advantage of a specific completion, its reward is compared against the average reward of the *other* $k-1$ completions in the same batch.
* **Efficiency:** This allows the model to learn from both good and bad generations simultaneously within a single update step.

## 🛠️ Installation

This project relies heavily on the Hugging Face `trl` library.

**Note:** To ensure compatibility with manual training loops (specifically the `.step()` method needed for granular control), this project uses a specific stable version of `trl`.

```bash
# 1. Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# 2. Install dependencies
pip install torch transformers datasets accelerate numpy tqdm matplotlib seaborn
pip install trl==0.8.6 
