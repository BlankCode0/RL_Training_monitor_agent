"""
Shared utilities for the DPO Training Monitor project.
"""

import json
import yaml
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainingLogEntry:
    """Single training log entry with all tracked metrics."""
    step: int
    epoch: float
    loss: float
    learning_rate: float
    
    # DPO-specific metrics
    rewards_chosen: float        # Mean reward for chosen responses
    rewards_rejected: float      # Mean reward for rejected responses
    reward_margin: float         # chosen - rejected (should be positive and stable)
    reward_accuracy: float       # % of times chosen > rejected
    
    # Optional eval metrics (logged less frequently)
    eval_loss: Optional[float] = None
    eval_reward_accuracy: Optional[float] = None
    eval_reward_margin: Optional[float] = None
    
    timestamp: Optional[str] = None


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def append_log(log_entry: TrainingLogEntry, log_file: str):
    """Append a training log entry to JSONL file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    entry = asdict(log_entry)
    entry['timestamp'] = datetime.now().isoformat()
    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def read_logs(log_file: str, last_n: Optional[int] = None) -> list[dict]:
    """Read training logs from JSONL file."""
    logs = []
    if not os.path.exists(log_file):
        return logs
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    if last_n is not None:
        logs = logs[-last_n:]
    return logs


def format_logs_for_agent(logs: list[dict]) -> str:
    """Format log entries into a readable string for the LLM agent."""
    if not logs:
        return "No log entries available."
    
    lines = []
    lines.append("=" * 80)
    lines.append("DPO TRAINING LOG")
    lines.append(f"Showing steps {logs[0]['step']} to {logs[-1]['step']}")
    lines.append("=" * 80)
    lines.append("")
    
    # Header
    lines.append(f"{'Step':>6} | {'Epoch':>5} | {'Loss':>8} | {'LR':>10} | "
                 f"{'Rew_Chosen':>10} | {'Rew_Reject':>10} | {'Margin':>8} | "
                 f"{'Rew_Acc':>7} | {'Eval_Loss':>9} | {'Eval_Acc':>8}")
    lines.append("-" * 110)
    
    for log in logs:
        eval_loss = f"{log.get('eval_loss', ''):<9.4f}" if log.get('eval_loss') is not None else "    -    "
        eval_acc = f"{log.get('eval_reward_accuracy', ''):<8.4f}" if log.get('eval_reward_accuracy') is not None else "   -    "
        
        lines.append(
            f"{log['step']:>6} | {log['epoch']:>5.2f} | {log['loss']:>8.4f} | "
            f"{log['learning_rate']:>10.2e} | {log['rewards_chosen']:>10.4f} | "
            f"{log['rewards_rejected']:>10.4f} | {log['reward_margin']:>8.4f} | "
            f"{log['reward_accuracy']:>7.2%} | {eval_loss} | {eval_acc}"
        )
    
    lines.append("")
    
    # Compute trends for last few entries
    if len(logs) >= 5:
        recent = logs[-5:]
        losses = [l['loss'] for l in recent]
        margins = [l['reward_margin'] for l in recent]
        accs = [l['reward_accuracy'] for l in recent]
        
        loss_trend = losses[-1] - losses[0]
        margin_trend = margins[-1] - margins[0]
        acc_trend = accs[-1] - accs[0]
        
        lines.append("RECENT TRENDS (last 5 entries):")
        lines.append(f"  Loss change:   {loss_trend:+.4f} ({'decreasing ✓' if loss_trend < 0 else 'increasing ✗'})")
        lines.append(f"  Margin change: {margin_trend:+.4f} ({'increasing' if margin_trend > 0 else 'decreasing'})")
        lines.append(f"  Accuracy change: {acc_trend:+.4f}")
        
        # Check for potential issues
        if len(logs) >= 10:
            eval_losses = [l['eval_loss'] for l in logs if l.get('eval_loss') is not None]
            if len(eval_losses) >= 3:
                if eval_losses[-1] > eval_losses[-2] > eval_losses[-3]:
                    lines.append("  ⚠️  Eval loss increasing for 3 consecutive checks")
                if all(l['reward_accuracy'] > 0.95 for l in logs[-5:]):
                    lines.append("  ⚠️  Reward accuracy saturated above 95%")
    
    return '\n'.join(lines)


def save_results(results: dict, filepath: str):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")