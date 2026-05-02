# LLM-Based DPO Training Monitor Agent

**Raj Christ Ronghang** | M.Tech Artificial Intelligence | IIT Kharagpur  
**Thesis Guide:** Prof. Prabhat Kumar Mishra

---

## Overview

This repository contains the implementation of an **LLM-based monitoring agent** for Direct Preference Optimization (DPO) training. Instead of relying on hand-tuned heuristic rules to decide when to stop training, this system uses **Llama 3.3 70B** (via Groq API) to read training logs in real time, reason over multiple signals simultaneously, and autonomously stop training when it detects failure modes such as overfitting, reward hacking, or training collapse.

The agent was evaluated against **12 heuristic baselines** across two model architectures — **TinyLlama 1.1B** and **SmolLM2 1.7B** — demonstrating consistent detection performance without any manual threshold tuning.

---

## Repository Structure

```
thesis-BlankCode0/
├── dpo_agent_monitor/        # Experiment 1 — TinyLlama 1.1B
│   ├── src/
│   │   ├── agent.py          # LLM monitoring agent
│   │   ├── train.py          # DPO training script
│   │   ├── baselines.py      # 12 heuristic baseline methods
│   │   ├── evaluate_quality.py
│   │   └── utils.py
│   ├── configs/
│   ├── logs/
│   ├── requirements.txt
│   └── README.md             # Detailed run instructions
│
├── dpo_agent_monitor2/       # Experiment 2 — SmolLM2 1.7B
│   ├── src/
│   │   ├── agent.py
│   │   ├── train.py
│   │   ├── baselines.py
│   │   ├── evaluate_quality.py
│   │   └── utils.py
│   ├── configs/
│   ├── logs/
│   ├── requirements.txt
│   └── README.md             # Detailed run instructions
│
└── README.md                 # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- Groq API key — get one free at [console.groq.com](https://console.groq.com)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/RHPILab/thesis-BlankCode0.git
cd thesis-BlankCode0
```

### Step 2 — Set Up Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

For Experiment 1 (TinyLlama):
```bash
cd dpo_agent_monitor
pip install -r requirements.txt
```

For Experiment 2 (SmolLM2):
```bash
cd dpo_agent_monitor2
pip install -r requirements.txt
```

### Step 4 — Set Groq API Key

```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

To make it permanent, add the above line to your `~/.bashrc` or `~/.zshrc`.

> See individual folder `README.md` files for detailed instructions on running training, the monitoring agent, baselines, and evaluation.

---

## How It Works

### The Problem

Standard DPO fine-tuning can silently fail in three ways:

| Failure Mode | What Happens |
|---|---|
| **Overfitting** | Train loss drops, eval loss rises — model memorises training preferences |
| **Reward Hacking** | Model exploits reward signal — margin spikes, rewards drift to extremes |
| **Training Collapse** | Loss becomes NaN, gradients explode — model weights corrupted |

Catching these manually requires constant monitoring or brittle hand-tuned rules that need re-calibration for every new model architecture.

### The Solution

An LLM agent reads training logs every 50 steps and reasons over **four signals simultaneously**:

```
1. Training loss        — is the model still learning?
2. Eval loss            — is the model generalising or overfitting?
3. Reward margin        — is the preference gap healthy or exploding?
4. Reward accuracy      — is the model distinguishing chosen from rejected?
```

The agent returns a structured decision — `healthy`, `warning`, or `stop` — with a confidence score, reasoning, and detected issues. When confidence crosses **0.70** with status `stop`, a stop file is written and training halts cleanly at that checkpoint.

### System Architecture

```
Training Script (train.py)
    │
    ├── writes metrics every step → training_logs.jsonl
    │
    └── checks for STOP_TRAINING file every step
              ↑
              │ writes stop file if confidence > 0.70
              │
    Monitoring Agent (agent.py)
        │
        ├── polls log file every 10 seconds
        ├── takes last 20 log entries as window
        ├── pre-computes statistics (velocities, consecutive counts)
        └── sends to Llama 3.3 70B via Groq API → get decision
```

---

## What the Agent Detects

### Overfitting
- Eval loss increasing for **3 or more consecutive** eval checkpoints
- Eval loss rising more than **0.15 above its minimum** observed value
- Severe train-eval gap: train loss **< 0.15** while eval loss **> 0.75**

### Reward Hacking (Detection of reward Hacking is lacking)
- Reward margin exceeding **3.0** (absolute threshold)
- Margin velocity **≥ 1.0** over last 10 steps (rate-of-change detection)
- Chosen and rejected rewards flying apart in **opposite directions** rapidly
- Reward accuracy sustained **above 0.93** for 3+ consecutive entries

### Training Collapse
- Loss becomes **NaN**
- Both chosen and rejected rewards drift **below -3.0 or above +3.0**
- Rewards collapse to **identical values** (model stops distinguishing)

### Anti-Hallucination Design
Pre-computed statistics (consecutive eval loss increases, margin velocity, accuracy counts) are injected directly into the prompt as verified numbers. The agent is instructed to only flag an issue if it has **actually crossed its defined threshold**, and must cite the exact value and threshold for every detected issue.

---

## Results

### TinyLlama 1.1B

| Method | Stop Step | Eval Loss at Stop |
|---|---|---|
| No stopping | — | 0.867 |
| Eval loss patience (k=3) | 350 | 0.672 |
| Reward margin threshold (3.0) | 900 | 0.798 |
| Combined heuristic | 350 | 0.672 |
| **LLM Agent (ours)** | **350** | **0.672** |
| Optimal (hindsight) | 250 | 0.615 |

### SmolLM2 1.7B

| Method | Stop Step | Eval Loss at Stop |
|---|---|---|
| No stopping | — | 0.741 |
| Single-metric baselines (×7) | Never | — |
| Combined heuristic | 650 | 0.633 |
| **LLM Agent (ours)** | **625** | **0.631** |
| Optimal (hindsight) | 300 | 0.624 |

**Key findings:**
- The agent matched the best heuristic baseline on TinyLlama with zero threshold tuning
- The agent outperformed the combined heuristic by 25 steps on SmolLM2
- 7 out of 12 single-metric baselines failed entirely on SmolLM2 — the agent did not
- The same agent prompt generalised across both architectures without modification
- Multi-signal detection confirmed at step 900: rewards below -3.0, accuracy above 0.93, and train-eval gap all triggered simultaneously

---

## Limitations

- **Catastrophic forgetting not detected** — if DPO causes the model to lose general knowledge, training logs alone won't show it. External benchmarks (MMLU, HellaSwag) would need to be run during training.

- **Mode collapse not detected** — if the model starts producing narrow, repetitive responses, diversity metrics are needed beyond loss and reward signals.

- **KL divergence not tracked** — divergence from the reference model is not currently logged. Adding it to the JSONL log would give the agent an additional alignment signal.

- **Context window dependency** — the agent reasons over a fixed 20-step window. Signals that develop slowly across many steps may not be visible within a single window. Events seen in one window are not remembered in the next.

- **API latency** — the monitoring agent depends on the Groq API. Network issues or rate limits could delay or interrupt monitoring during long training runs.

- **Threshold descriptions still required** — while no numerical thresholds need to be tuned per model, the natural language description of failure modes in the system prompt was written for DPO specifically. Other fine-tuning methods may need prompt modifications.

---

## Future Work

- **Adaptive intervention** — instead of only stopping training, the agent could reduce the learning rate, adjust beta, or trigger a checkpoint rollback when warning-level signals appear. This turns a monitor into a controller.

- **Velocity-based early warning** — adding rate-of-change analysis (margin velocity, chosen/rejected divergence velocity) would allow the agent to detect reward hacking during the acceleration phase, before rewards reach extreme values.

- **Persistent context across windows** — maintaining a summary of worst-case metrics seen across all previous windows (not just the current 20-step window) would prevent important signals from disappearing as the window slides forward.

- **Validation at larger scales** — experiments were conducted on 1.1B and 1.7B models due to hardware constraints. Validating on 7B and 70B models is the natural next step.

- **Multi-objective alignment** — extending the agent to monitor simultaneous alignment objectives (helpfulness + safety + honesty) with separate reward signals per dimension.

- **Distilled monitor model** — training a small (1B) monitor-specific model on agent decisions from many training runs, eliminating the API dependency and enabling fully local deployment.

---

## Models Used

| Role | Model | Access |
|---|---|---|
| Policy model (Experiment 1) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Hugging Face |
| Policy model (Experiment 2) | HuggingFaceTB/SmolLM2-1.7B-Instruct | Hugging Face |
| Monitoring agent | Llama 3.3 70B Versatile | Groq API |
| Fine-tuning method | DPO via Hugging Face TRL | — |
| Parameter-efficient training | LoRA (rank=16, alpha=32) | PEFT |
| Preference dataset | UltraFeedback (openbmb/UltraFeedback) | Hugging Face |

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{ronghang2026llm,
  author    = {Raj Christ Ronghang},
  title     = {Reinforcement Fine-Tuning of Large Language Models with Automated Training Monitoring},
  school    = {Indian Institute of Technology Kharagpur},
  year      = {2026},
  type      = {M.Tech Thesis},
  advisor   = {Prof. Prabhat Kumar Mishra}
}
```

---

## Acknowledgements

This work was carried out under the guidance of **Prof. Prabhat Kumar Mishra** at IIT Kharagpur. Compute support provided by the AI GPU Lab, IIT Kharagpur.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
    
