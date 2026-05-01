"""
Heuristic Baseline Stopping Methods

These are the baselines we compare our LLM agent against.
Each implements a different simple strategy for deciding when to stop DPO training.
"""

import argparse
import json
from utils import read_logs, load_config


class BaselineResult:
    def __init__(self, name: str, stop_step: int | None, reason: str):
        self.name = name
        self.stop_step = stop_step  # None means "never stopped"
        self.reason = reason
    
    def to_dict(self):
        return {
            "baseline": self.name,
            "stop_step": self.stop_step,
            "reason": self.reason,
        }


def baseline_no_stopping(logs: list[dict]) -> BaselineResult:
    """
    Baseline 1: No early stopping at all.
    Train for the full duration. This is the "do nothing" baseline.
    """
    return BaselineResult(
        name="no_stopping",
        stop_step=None,
        reason="No early stopping applied — trained to completion",
    )


def baseline_eval_loss_patience(logs: list[dict], patience: int = 3) -> BaselineResult:
    """
    Baseline 2: Stop when eval loss increases for `patience` consecutive eval checks.
    This is the most common early stopping heuristic.
    """
    eval_entries = [(l['step'], l['eval_loss']) for l in logs if l.get('eval_loss') is not None]
    
    if len(eval_entries) < 2:
        return BaselineResult("eval_loss_patience", None, "Not enough eval data")
    
    best_eval_loss = float('inf')
    patience_counter = 0
    
    for step, eval_loss in eval_entries:
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                return BaselineResult(
                    name=f"eval_loss_patience_{patience}",
                    stop_step=step,
                    reason=f"Eval loss increased for {patience} consecutive checks. "
                           f"Best eval loss: {best_eval_loss:.4f}, current: {eval_loss:.4f}",
                )
    
    return BaselineResult(
        f"eval_loss_patience_{patience}", None,
        f"Eval loss never increased for {patience} consecutive checks",
    )


def baseline_reward_margin_threshold(logs: list[dict], threshold: float = 5.0) -> BaselineResult:
    """
    Baseline 3: Stop when reward margin exceeds a threshold.
    Large margins indicate potential reward hacking.
    """
    for log in logs:
        if abs(log['reward_margin']) > threshold:
            return BaselineResult(
                name=f"margin_threshold_{threshold}",
                stop_step=log['step'],
                reason=f"Reward margin {log['reward_margin']:.4f} exceeded threshold {threshold}",
            )
    
    return BaselineResult(
        f"margin_threshold_{threshold}", None,
        f"Reward margin never exceeded {threshold}",
    )


def baseline_reward_accuracy_saturation(logs: list[dict], 
                                         threshold: float = 0.95,
                                         sustained_steps: int = 5) -> BaselineResult:
    """
    Baseline 4: Stop when reward accuracy is saturated above threshold
    for a sustained number of logging steps.
    Near-100% accuracy often indicates reward hacking.
    """
    counter = 0
    for log in logs:
        if log['reward_accuracy'] >= threshold:
            counter += 1
            if counter >= sustained_steps:
                return BaselineResult(
                    name=f"acc_saturation_{threshold}",
                    stop_step=log['step'],
                    reason=f"Reward accuracy >= {threshold} for {sustained_steps} consecutive entries",
                )
        else:
            counter = 0
    
    return BaselineResult(
        f"acc_saturation_{threshold}", None,
        f"Reward accuracy never saturated at {threshold} for {sustained_steps} steps",
    )


def baseline_loss_plateau(logs: list[dict], 
                           window: int = 10, 
                           min_improvement: float = 0.001) -> BaselineResult:
    """
    Baseline 5: Stop when training loss plateaus.
    If loss doesn't improve by min_improvement over a window of steps.
    """
    # Filter out eval-only rows (loss=0.0)
    train_logs = [l for l in logs if l['loss'] > 0]
    
    if len(train_logs) < window:
        return BaselineResult("loss_plateau", None, "Not enough data")
    
    for i in range(window, len(train_logs)):
        window_logs = train_logs[i-window:i]
        loss_change = window_logs[0]['loss'] - window_logs[-1]['loss']
        
        if loss_change < min_improvement:
            return BaselineResult(
                name=f"loss_plateau_w{window}",
                stop_step=train_logs[i]['step'],
                reason=f"Loss improvement over last {window} entries: {loss_change:.6f} "
                       f"(below threshold {min_improvement})",
            )
    
    return BaselineResult(
        f"loss_plateau_w{window}", None,
        "Loss never plateaued",
    )


def baseline_train_eval_divergence(logs: list[dict], 
                                    max_gap: float = 0.5) -> BaselineResult:
    """
    Baseline 6: Stop when train and eval loss diverge significantly.
    A large gap indicates overfitting.
    """
    # Get the most recent train loss at each eval checkpoint
    last_train_loss = None
    for log in logs:
        # Skip eval-only rows (loss=0.0 with eval_loss present)
        if log['loss'] > 0:
            last_train_loss = log['loss']
        
        if log.get('eval_loss') is not None and log['eval_loss'] > 0 and last_train_loss is not None:
            gap = log['eval_loss'] - last_train_loss
            if gap > max_gap:
                return BaselineResult(
                    name=f"train_eval_divergence_{max_gap}",
                    stop_step=log['step'],
                    reason=f"Train-eval loss gap: {gap:.4f} exceeded threshold {max_gap}. "
                           f"Train loss: {last_train_loss:.4f}, Eval loss: {log['eval_loss']:.4f}",
                )
    
    return BaselineResult(
        f"train_eval_divergence_{max_gap}", None,
        f"Train-eval divergence never exceeded {max_gap}",
    )


def baseline_combined_heuristic(logs: list[dict],
                                 min_steps: int = 200) -> BaselineResult:
    """
    Baseline 7: Combined multi-signal heuristic.
    This mimics what an experienced ML engineer would check manually —
    looking at multiple signals together rather than any single metric.
    
    Stops when AT LEAST 2 of these conditions are true simultaneously:
    1. Eval loss has increased for 2+ consecutive checks
    2. Reward accuracy > 0.85
    3. Reward margin > 2.0
    4. Train-eval loss gap > 0.2
    
    Only activates after min_steps to avoid false positives during warmup.
    """
    # Track eval loss trend
    eval_entries = []
    last_train_loss = None
    eval_increasing_count = 0
    best_eval_loss = float('inf')
    
    for log in logs:
        # Skip early training
        if log['step'] < min_steps:
            if log['loss'] > 0:
                last_train_loss = log['loss']
            if log.get('eval_loss') is not None and log['eval_loss'] > 0:
                best_eval_loss = min(best_eval_loss, log['eval_loss'])
            continue
        
        # Track train loss (skip eval-only rows)
        if log['loss'] > 0:
            last_train_loss = log['loss']
        
        # On eval checkpoints, check all signals
        if log.get('eval_loss') is not None and log['eval_loss'] > 0:
            current_eval = log['eval_loss']
            
            if current_eval < best_eval_loss:
                best_eval_loss = current_eval
                eval_increasing_count = 0
            else:
                eval_increasing_count += 1
            
            eval_entries.append(current_eval)
            
            # Count how many conditions are met
            conditions_met = []
            
            # Condition 1: Eval loss rising
            if eval_increasing_count >= 2:
                conditions_met.append(f"eval_loss rising for {eval_increasing_count} checks")
            
            # Condition 2: High reward accuracy (check last training log)
            recent_train = [l for l in logs if l['step'] <= log['step'] and l['loss'] > 0]
            if recent_train:
                latest = recent_train[-1]
                if latest['reward_accuracy'] > 0.85:
                    conditions_met.append(f"reward_acc={latest['reward_accuracy']:.3f} > 0.85")
                
                # Condition 3: High reward margin
                if latest['reward_margin'] > 2.0:
                    conditions_met.append(f"margin={latest['reward_margin']:.3f} > 2.0")
                
                # Condition 4: Train-eval gap
                if last_train_loss is not None:
                    gap = current_eval - last_train_loss
                    if gap > 0.2:
                        conditions_met.append(f"train-eval gap={gap:.3f} > 0.2")
            
            # Stop if 2+ conditions met
            if len(conditions_met) >= 2:
                return BaselineResult(
                    name="combined_heuristic",
                    stop_step=log['step'],
                    reason=f"Multiple signals triggered: {'; '.join(conditions_met)}",
                )
    
    return BaselineResult(
        "combined_heuristic", None,
        "Combined conditions never met simultaneously",
    )


def baseline_ema_eval_loss(logs: list[dict], 
                            alpha: float = 0.3,
                            patience: int = 3) -> BaselineResult:
    """
    Baseline 8: Exponential Moving Average of eval loss.
    More robust than raw eval loss — smooths out noise.
    
    Stops when the EMA of eval loss has been rising for `patience` 
    consecutive eval checks.
    """
    eval_entries = [(l['step'], l['eval_loss']) 
                    for l in logs 
                    if l.get('eval_loss') is not None and l['eval_loss'] > 0]
    
    if len(eval_entries) < 3:
        return BaselineResult("ema_eval_loss", None, "Not enough eval data")
    
    # Compute EMA
    ema = eval_entries[0][1]  # Initialize with first value
    ema_values = [(eval_entries[0][0], ema)]
    
    for step, val in eval_entries[1:]:
        ema = alpha * val + (1 - alpha) * ema
        ema_values.append((step, ema))
    
    # Check for sustained increase in EMA
    increasing_count = 0
    for i in range(1, len(ema_values)):
        if ema_values[i][1] > ema_values[i-1][1]:
            increasing_count += 1
            if increasing_count >= patience:
                return BaselineResult(
                    name=f"ema_eval_loss_a{alpha}_p{patience}",
                    stop_step=ema_values[i][0],
                    reason=f"EMA of eval loss rising for {patience} consecutive checks. "
                           f"EMA: {ema_values[i-patience][1]:.4f} → {ema_values[i][1]:.4f}",
                )
        else:
            increasing_count = 0
    
    return BaselineResult(
        f"ema_eval_loss_a{alpha}_p{patience}", None,
        "EMA of eval loss never showed sustained increase",
    )


def run_all_baselines(log_file: str) -> list[dict]:
    """Run all baseline methods on the logs and return results."""
    logs = read_logs(log_file)
    
    if not logs:
        print(f"No logs found in {log_file}")
        return []
    
    print(f"Running baselines on {len(logs)} log entries")
    print(f"Steps: {logs[0]['step']} to {logs[-1]['step']}")
    print("=" * 70)
    
    baselines = [
        baseline_no_stopping(logs),
        baseline_eval_loss_patience(logs, patience=3),
        baseline_eval_loss_patience(logs, patience=5),
        baseline_reward_margin_threshold(logs, threshold=3.0),
        baseline_reward_margin_threshold(logs, threshold=5.0),
        baseline_reward_accuracy_saturation(logs, threshold=0.95, sustained_steps=5),
        baseline_reward_accuracy_saturation(logs, threshold=0.98, sustained_steps=3),
        baseline_loss_plateau(logs, window=10),
        baseline_train_eval_divergence(logs, max_gap=0.3),
        baseline_train_eval_divergence(logs, max_gap=0.5),
        baseline_combined_heuristic(logs),
        baseline_ema_eval_loss(logs, alpha=0.3, patience=3),
        baseline_ema_eval_loss(logs, alpha=0.5, patience=3),
    ]
    
    results = []
    for b in baselines:
        stop_str = f"Step {b.stop_step}" if b.stop_step else "Never"
        print(f"  {b.name:<35s} → Stop: {stop_str:<12s} | {b.reason}")
        results.append(b.to_dict())
    
    # Save results
    output_file = log_file.replace('.jsonl', '_baseline_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline stopping methods")
    parser.add_argument("--log_file", type=str, required=True)
    args = parser.parse_args()
    
    run_all_baselines(args.log_file)