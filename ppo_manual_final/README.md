---
base_model: EleutherAI/pythia-410m-deduped
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:EleutherAI/pythia-410m-deduped
- lora
- transformers
---

# 🎬 PPO: Custom PyTorch Implementation for LLM Alignment

This repository contains a **manual implementation "from scratch" of the Proximal Policy Optimization (PPO)** algorithm using pure PyTorch.

The goal is to align a **Pythia-410m** language model to generate **positive movie reviews** (IMDB dataset).

## 🚀 Key Features

Unlike standard implementations using high-level libraries like `trl.PPOTrainer`, this project implements the PPO training loop manually to allow granular control:

* **⚡ Pure PyTorch Loop:** Full control over Rollout, Advantage Calculation (GAE), and Optimization steps.
* **🛠️ Technical Fixes:**
    * **Sequence-Level Value Estimation:** Adapts the Critic's scoring to handle sequence-level rewards (DistilBERT) correctly avoiding dimension mismatch errors.
* **📊 Real-time Tracking:** Integrated with **Weights & Biases (WandB)** for live metric monitoring.

## 📈 Training Metrics (WandB)

You can visualize the training curves, including the Reward evolution and KL Divergence stability, on the WandB dashboard:

[![WandB](https://img.shields.io/badge/WandB-Log-orange?style=for-the-badge&logo=weightsandbiases)](https://wandb.ai/charlene-krick-ensta-paris/ppo-manual-pytorch/runs/x736odmg?nw=nwusercharlenekrick)

👉 **[Click here to access the Live Dashboard](https://wandb.ai/charlene-krick-ensta-paris/ppo-manual-pytorch/runs/x736odmg?nw=nwusercharlenekrick)**

## 🧠 Architecture

* **Actor (Policy):** `EleutherAI/pythia-410m-deduped` (Fine-tuned with LoRA).
* **Reference Model:** Frozen copy of the SFT model (to compute KL Divergence).
* **Critic (Value Model):** `Pythia-410m` with a scalar head (LoRA), trained to predict the final reward.
* **Reward Model (Judge):** `lvwerra/distilbert-imdb` (External Sentiment Classifier).

## ⚙️ Hyperparameters & Configuration

To ensure a fair comparison with RLOO, the following strict configuration was used:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **KL Beta** | `0.05` | Constraint strength (Adaptive KL off) |
| **Learning Rate** | `3e-5` | Cosine Scheduler |
| **Batch Size** | `32` | Achieved via Gradient Accumulation (2 * 16) |
| **Max New Tokens** | `48` | Generation length |
| **Max Steps** | `100` | Training duration |
| **PPO Clip** | `0.2` | Standard clipping epsilon |

### 🏆 Reward Function
The model is trained on a **Composite Reward** function to enforce quality and style:

$$R_{total} = R_{positive} + 0.5 \times R_{repetition\_penalty} + R_{length\_penalty}$$

* **Positive Score:** Probability from DistilBERT.
* **Repetition Penalty:** Penalizes loops and repetitive patterns.
* **Length Penalty:** Applies a `-2.0` penalty if the generation is shorter than 5 words.
* **Text Cleaning:** Inputs are cleaned (HTML tags removal) before scoring.

## 🛠️ Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/REPO_NAME.git](https://github.com/YOUR_USERNAME/REPO_NAME.git)
   cd REPO_NAME

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.18.0
