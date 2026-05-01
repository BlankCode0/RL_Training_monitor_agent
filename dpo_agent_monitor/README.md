# DPO Training Monitor Agent

## An LLM-Based Agent for Detecting Reward Hacking and Overfitting in DPO Training

### MTP Project — IIT Kharagpur

---

## Overview

This project investigates whether an LLM-based agent can effectively monitor Direct Preference Optimization (DPO) training logs in real-time and detect failure modes such as **reward hacking**, **overfitting** and model **collapse** comparing its performance against heuristic baselines.

## Project Structure

```
dpo-agent-monitor/
├── configs/
│   └── training_config.yaml       # DPO training hyperparameters
├── src/
│   ├── train_dpo.py               # DPO training script with logging
│   ├── agent.py                   # LLM-based monitoring agent
│   ├── monitor.py                 # Real-time log monitoring + agent integration
│   ├── baselines.py               # Heuristic baseline stopping methods
│   ├── evaluate.py                # Evaluate model quality at different stop points
│   └── utils.py                   # Shared utilities
├── scripts/
│   └── analyze_results.py         # Generate plots and comparison tables
├── logs/                          # Training logs (auto-generated)
├── results/                       # Experiment results and plots
└── README.md
```

## Setup

```bash
pip install torch transformers trl datasets peft accelerate wandb openai pyyaml --break-system-packages
```

## Groq API Setup

The agent uses Groq's free API (OpenAI-compatible):
```bash
export GROQ_API_KEY="your-key-from-console.groq.com"
```

Available free models on Groq:
- `llama-3.3-70b-versatile` — Best reasoning, 12K output tokens (recommended)
- `meta-llama/llama-4-scout-17b-16e-instruct` — 30K output, good for detailed analysis
- `qwen/qwen3-32b` — Strong alternative
- `moonshotai/kimi-k2-instruct` — 10K output, 1M context window

## Usage

```bash
# Step 1: Run DPO training with full logging
python src/train_dpo.py --config configs/training_config.yaml

# Step 2: Test agent on collected logs (offline)
python src/agent.py --log_file logs/training_run_1.jsonl --mode offline

# Step 3: Run with live monitoring
python src/monitor.py --config configs/training_config.yaml

# Step 4: Run baselines for comparison
python src/baselines.py --log_file logs/training_run_1.jsonl

# Step 5: Evaluate and compare
python scripts/analyze_results.py --log_file logs/training_run.jsonl --output_dir results/
```