# MiniRLHF: RLHF for Small Language Models (GPT-2)

MiniRLHF explores how Reinforcement Learning from Human Feedback (RLHF) can improve the behavior of small language models.  
The project fine‑tunes GPT‑2 on a summarization task and then applies Proximal Policy Optimization (PPO) to align the model’s outputs with human‑preferred summaries.  
The goal is to build a lightweight, fully reproducible RLHF pipeline that mirrors the alignment process used in larger systems.

## Project Overview

- **Task:** Summarization of Reddit posts using the OpenAI TL;DR dataset  
- **Model:** GPT‑2 (124M or 355M) fine‑tuned for summarization  
- **Pipeline:**  
  - **Supervised Fine‑Tuning (SFT):** trains a baseline GPT‑2 summarizer  
  - **Reward Model:** scores summaries using a preference‑based reward signal  
  - **RLHF (PPO):** optimizes the model to generate more helpful, concise summaries  
- **Goal:** Produce summaries that better match human‑preferred behavior compared to the purely supervised model

## Repository Structure

```
data/                # Dataset loading and preprocessing
scripts/             # Training and evaluation scripts
    train_sft.py
    train_reward_model.py
    train_rl.py
    evaluate.py
models/              # Saved checkpoints
results/             # Metrics and example outputs
requirements.txt     # Package dependencies
README.md            # Project description
```

## Usage

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Dataset
The OpenAI TL;DR dataset will automatically download through the Hugging Face datasets library.

## Training

### 1. Supervised Fine‑Tuning (SFT)
```
python scripts/train_sft.py     --model_name gpt2     --dataset OpenAI/summarize_tldr     --output_dir outputs/sft_model
```

### 2. Reward Model
The project uses a pre‑trained summarization reward model:

OpenAssistant/reward-model-deberta-v3-large-v2

### 3. RLHF (PPO Training)
```
python scripts/train_rl.py     --sft_model outputs/sft_model     --reward_model OpenAssistant/reward-model-deberta-v3-large-v2     --output_dir outputs/rlhf_model
```

PPO training updates the SFT model toward higher‑reward summaries.  
Checkpoints and logs are saved automatically.

## Evaluation
```
python scripts/evaluate.py     --model_dir outputs/rlhf_model     --baseline_dir outputs/sft_model
```

Evaluation includes:
- Summaries from both models  
- Reward model scores  
- ROUGE metrics  
- Saved qualitative examples in the `results/` directory  

## Results

As training progresses, this section will include:
- Reward score comparisons between SFT and RLHF models  
- Example summaries showing qualitative differences  
- Plots and analysis generated during evaluation  

## References

- Stiennon et al., *Learning to Summarize with Human Feedback* (OpenAI, 2020)  
- Ziegler et al., *Fine‑Tuning Language Models from Human Preferences* (2019)  
- Hugging Face TRL Library (PPO implementation)  
- OpenAssistant Reward Model  
