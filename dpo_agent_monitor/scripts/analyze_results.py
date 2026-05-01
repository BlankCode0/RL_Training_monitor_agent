"""
Results Analysis and Visualization

Generates plots and comparison tables for the MTP report.
Compares agent stopping decisions against baselines.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils import read_logs


def plot_training_curves(log_file: str, output_dir: str, 
                          agent_stop: int = None, baseline_stops: dict = None):
    """
    Plot training curves with stop points marked.
    
    Args:
        log_file: Path to training log JSONL
        output_dir: Directory to save plots
        agent_stop: Step where agent recommended stopping
        baseline_stops: Dict of {baseline_name: stop_step}
    """
    logs = read_logs(log_file)
    if not logs:
        print("No logs to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    steps = [l['step'] for l in logs]
    losses = [l['loss'] for l in logs]
    margins = [l['reward_margin'] for l in logs]
    accuracies = [l['reward_accuracy'] for l in logs]
    rew_chosen = [l['rewards_chosen'] for l in logs]
    rew_rejected = [l['rewards_rejected'] for l in logs]
    
    # Eval metrics (sparse)
    eval_steps = [l['step'] for l in logs if l.get('eval_loss') is not None]
    eval_losses = [l['eval_loss'] for l in logs if l.get('eval_loss') is not None]
    eval_accs = [l.get('eval_reward_accuracy') for l in logs if l.get('eval_loss') is not None]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DPO Training Dynamics with Stop Points', fontsize=14, fontweight='bold')
    
    colors = {
        'agent': '#e74c3c',
        'eval_loss_patience_3': '#3498db',
        'margin_threshold_5.0': '#2ecc71',
        'acc_saturation_0.95': '#9b59b6',
        'loss_plateau_w10': '#f39c12',
    }
    
    def add_stop_lines(ax, agent_stop, baseline_stops):
        """Add vertical lines for stop points."""
        if agent_stop:
            ax.axvline(x=agent_stop, color=colors['agent'], linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'Agent (step {agent_stop})')
        if baseline_stops:
            for name, step in baseline_stops.items():
                if step and name in colors:
                    ax.axvline(x=step, color=colors[name], linestyle=':', 
                               linewidth=1.5, alpha=0.6, label=f'{name} ({step})')
    
    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(steps, losses, 'b-', alpha=0.7, label='Train Loss')
    if eval_steps:
        ax.plot(eval_steps, eval_losses, 'r-o', markersize=4, label='Eval Loss')
    add_stop_lines(ax, agent_stop, baseline_stops)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Eval Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 2. Reward Margin
    ax = axes[0, 1]
    ax.plot(steps, margins, 'g-', alpha=0.7)
    add_stop_lines(ax, agent_stop, baseline_stops)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward Margin')
    ax.set_title('Reward Margin (Chosen - Rejected)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 3. Reward Accuracy
    ax = axes[0, 2]
    ax.plot(steps, accuracies, 'm-', alpha=0.7, label='Train')
    if eval_accs and any(a is not None for a in eval_accs):
        valid_eval_accs = [(s, a) for s, a in zip(eval_steps, eval_accs) if a is not None]
        if valid_eval_accs:
            ax.plot([s for s, a in valid_eval_accs], 
                    [a for s, a in valid_eval_accs], 
                    'r-o', markersize=4, label='Eval')
    add_stop_lines(ax, agent_stop, baseline_stops)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reward Accuracy')
    ax.axhline(y=0.95, color='orange', linestyle=':', alpha=0.5, label='Saturation line')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 4. Individual Rewards
    ax = axes[1, 0]
    ax.plot(steps, rew_chosen, 'g-', alpha=0.7, label='Chosen')
    ax.plot(steps, rew_rejected, 'r-', alpha=0.7, label='Rejected')
    add_stop_lines(ax, agent_stop, baseline_stops)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Chosen vs Rejected Rewards')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 5. Train-Eval Gap
    ax = axes[1, 1]
    if eval_steps and eval_losses:
        # Interpolate train loss at eval steps
        train_at_eval = []
        for es in eval_steps:
            closest = min(logs, key=lambda l: abs(l['step'] - es))
            train_at_eval.append(closest['loss'])
        gaps = [e - t for e, t in zip(eval_losses, train_at_eval)]
        ax.plot(eval_steps, gaps, 'orange', marker='o', markersize=4)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=0.3, color='red', linestyle=':', alpha=0.5, label='Divergence threshold')
    add_stop_lines(ax, agent_stop, baseline_stops)
    ax.set_xlabel('Step')
    ax.set_ylabel('Eval Loss - Train Loss')
    ax.set_title('Train-Eval Divergence')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 6. Agent Confidence over time (if available)
    ax = axes[1, 2]
    assessment_file = log_file.replace('.jsonl', '_agent_assessments.json')
    if os.path.exists(assessment_file):
        with open(assessment_file, 'r') as f:
            assessments = json.load(f)
        a_steps = [a['step'] for a in assessments]
        a_confidence = [a['confidence'] for a in assessments]
        status_colors = {'healthy': 'green', 'warning': 'orange', 'stop': 'red'}
        scatter_colors = [status_colors.get(a['status'], 'gray') for a in assessments]
        ax.scatter(a_steps, a_confidence, c=scatter_colors, s=60, zorder=5)
        ax.plot(a_steps, a_confidence, 'k-', alpha=0.3)
        
        # Legend
        patches = [mpatches.Patch(color=c, label=s) for s, c in status_colors.items()]
        ax.legend(handles=patches, fontsize=7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Agent Confidence')
    ax.set_title('Agent Status & Confidence')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {plot_path}")


def generate_comparison_table(log_file: str, output_dir: str):
    """
    Generate a comparison table of when each method would stop training.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load baseline results
    baseline_file = log_file.replace('.jsonl', '_baseline_results.json')
    agent_file = log_file.replace('.jsonl', '_agent_assessments.json')
    
    results = {}
    
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baselines = json.load(f)
        for b in baselines:
            results[b['baseline']] = {
                'stop_step': b['stop_step'],
                'reason': b['reason'],
            }
    
    if os.path.exists(agent_file):
        with open(agent_file, 'r') as f:
            assessments = json.load(f)
        # Find first stop recommendation
        for a in assessments:
            if a['status'] == 'stop':
                results['llm_agent'] = {
                    'stop_step': a['step'],
                    'reason': a['stop_reason'],
                }
                break
        if 'llm_agent' not in results:
            results['llm_agent'] = {
                'stop_step': None,
                'reason': 'Agent never recommended stopping',
            }
    
    # Print table
    print("\n" + "=" * 90)
    print("STOPPING METHOD COMPARISON")
    print("=" * 90)
    print(f"{'Method':<35} {'Stop Step':<12} {'Reason'}")
    print("-" * 90)
    
    for name, info in sorted(results.items()):
        step_str = str(info['stop_step']) if info['stop_step'] else "Never"
        reason_short = info['reason'][:50] + "..." if len(info['reason']) > 50 else info['reason']
        print(f"{name:<35} {step_str:<12} {reason_short}")
    
    print("=" * 90)
    
    # Save as JSON
    output_file = os.path.join(output_dir, 'comparison_table.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results")
    parser.add_argument("--log_file", type=str, default="logs/training_run.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()
    
    # Load stop points for plotting
    baseline_file = args.log_file.replace('.jsonl', '_baseline_results.json')
    agent_file = args.log_file.replace('.jsonl', '_agent_assessments.json')
    
    agent_stop = None
    baseline_stops = {}
    
    if os.path.exists(agent_file):
        with open(agent_file, 'r') as f:
            for a in json.load(f):
                if a['status'] == 'stop':
                    agent_stop = a['step']
                    break
    
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            for b in json.load(f):
                if b['stop_step']:
                    baseline_stops[b['baseline']] = b['stop_step']
    
    plot_training_curves(args.log_file, args.output_dir, agent_stop, baseline_stops)
    generate_comparison_table(args.log_file, args.output_dir)