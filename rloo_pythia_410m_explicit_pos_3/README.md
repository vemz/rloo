---
base_model: EleutherAI/pythia-410m-deduped
library_name: transformers
model_name: rloo_pythia_410m_explicit_pos_3
tags:
- generated_from_trainer
- trl
- rloo
licence: license
---

# Model Card for rloo_pythia_410m_explicit_pos_3

This model is a fine-tuned version of [EleutherAI/pythia-410m-deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/vemz-x/huggingface/runs/h4qfk5pc) 


This model was trained with RLOO, a method introduced in [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://huggingface.co/papers/2402.14740).

### Framework versions

- TRL: 0.26.2
- Transformers: 4.57.3
- Pytorch: 2.9.1
- Datasets: 4.4.2
- Tokenizers: 0.22.1

## Citations

Cite RLOO as:

```bibtex
@inproceedings{ahmadian2024back,
    title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
    author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {"{U}}st{"{u}}n and Sara Hooker},
    year         = 2024,
    booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
    pages        = {12248--12267},
    publisher    = {Association for Computational Linguistics},
    editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```