"""
LLM-Based DPO Training Monitor Agent

Uses Groq (via OpenAI-compatible API) to analyze DPO training logs
and detect reward hacking, overfitting, and other failure modes.

Improvements over v1:
  - Velocity-based reward hacking detection (rate of change, not just thresholds)
  - Chosen/rejected divergence pattern detection
  - Anti-hallucination: agent must cite exact values + threshold for every issue
  - Pre-computed statistics injected into prompt (removes guesswork from LLM)
  - Stricter output format enforcement
"""

import argparse
import json
import os
from openai import OpenAI
from dataclasses import dataclass
from enum import Enum
from utils import read_logs, format_logs_for_agent, load_config


class TrainingStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    STOP = "stop"


@dataclass
class AgentAssessment:
    """Structured output from the monitoring agent."""
    status: TrainingStatus
    confidence: float
    reasoning: str
    detected_issues: list
    recommendation: str
    stop_reason: str = ""


# ── KEY CHANGE 1: Pre-compute statistics before calling LLM ──────────────────
# This removes guesswork. The LLM sees exact numbers, not raw logs.
# It cannot hallucinate values it didn't compute itself.

def compute_window_stats(logs: list[dict]) -> dict:
    """
    Pre-compute key statistics from the log window.
    These are injected directly into the prompt so the LLM
    doesn't have to compute them itself (reducing hallucination).
    """
    # Separate eval rows from train rows
    eval_rows  = [l for l in logs if l.get('eval_loss') and l['eval_loss'] > 0]
    train_rows = [l for l in logs if l.get('loss') and l['loss'] > 0
                  and l.get('learning_rate', 0) > 0]

    stats = {}

    # ── Eval loss ────────────────────────────────────────────────────────────
    if eval_rows:
        eval_losses = [r['eval_loss'] for r in eval_rows]
        stats['eval_loss_values']    = eval_losses
        stats['eval_loss_min']       = round(min(eval_losses), 4)
        stats['eval_loss_latest']    = round(eval_losses[-1], 4)
        stats['eval_loss_rise']      = round(eval_losses[-1] - min(eval_losses), 4)
        # Count consecutive increases at the end of the list
        consec = 0
        for i in range(len(eval_losses) - 1, 0, -1):
            if eval_losses[i] > eval_losses[i - 1]:
                consec += 1
            else:
                break
        stats['eval_loss_consecutive_increases'] = consec

    # ── Train loss ───────────────────────────────────────────────────────────
    if train_rows:
        stats['train_loss_latest'] = round(train_rows[-1]['loss'], 4)

    # ── Train-eval gap ───────────────────────────────────────────────────────
    if eval_rows and train_rows:
        stats['train_eval_gap'] = round(
            eval_rows[-1]['eval_loss'] - train_rows[-1]['loss'], 4)

    # ── Reward margin ────────────────────────────────────────────────────────
    margin_rows = [l for l in logs if l.get('rewards/margins') is not None]
    if len(margin_rows) >= 2:
        margins = [r['rewards/margins'] for r in margin_rows]
        stats['margin_latest']   = round(margins[-1], 4)
        stats['margin_earliest'] = round(margins[0], 4)
        stats['margin_max']      = round(max(margins), 4)

        # Velocity: change over last 10 entries (or all if fewer)
        look_back = min(10, len(margins))
        stats['margin_velocity_last10'] = round(
            margins[-1] - margins[-look_back], 4)

        # Velocity: change over last 5 entries
        look_back5 = min(5, len(margins))
        stats['margin_velocity_last5'] = round(
            margins[-1] - margins[-look_back5], 4)

    # ── Reward accuracy ──────────────────────────────────────────────────────
    acc_rows = [l for l in logs if l.get('rewards/accuracies') is not None]
    if acc_rows:
        accs = [r['rewards/accuracies'] for r in acc_rows]
        stats['accuracy_latest']   = round(accs[-1], 4)
        stats['accuracy_earliest'] = round(accs[0], 4)
        # Count how many of the last 3 entries are above 0.93
        last3 = accs[-3:]
        stats['accuracy_above_093_last3'] = sum(1 for a in last3 if a > 0.93)
        stats['accuracy_last3_values']    = [round(a, 4) for a in last3]

    # ── Chosen / rejected rewards ────────────────────────────────────────────
    chosen_rows   = [l for l in logs if l.get('rewards/chosen')   is not None]
    rejected_rows = [l for l in logs if l.get('rewards/rejected') is not None]

    if chosen_rows and rejected_rows:
        chosen   = [r['rewards/chosen']   for r in chosen_rows]
        rejected = [r['rewards/rejected'] for r in rejected_rows]

        stats['chosen_latest']   = round(chosen[-1], 4)
        stats['rejected_latest'] = round(rejected[-1], 4)

        # Divergence velocity: chosen going up while rejected goes down
        if len(chosen) >= 5:
            stats['chosen_velocity_last5']   = round(chosen[-1]   - chosen[-5],   4)
            stats['rejected_velocity_last5'] = round(rejected[-1] - rejected[-5], 4)

    # ── Current step & epoch ─────────────────────────────────────────────────
    if logs:
        stats['current_step']  = logs[-1].get('step', 0)
        stats['current_epoch'] = round(logs[-1].get('epoch', 0), 3)

    return stats


def format_stats_for_prompt(stats: dict) -> str:
    """Format pre-computed stats into a clean block for the prompt."""
    lines = ["=== PRE-COMPUTED STATISTICS (use THESE values, do not re-compute) ==="]

    if 'current_step' in stats:
        lines.append(f"Current step  : {stats['current_step']}")
    if 'current_epoch' in stats:
        lines.append(f"Current epoch : {stats['current_epoch']}")

    lines.append("")
    lines.append("── EVAL LOSS ──")
    if 'eval_loss_values' in stats:
        lines.append(f"  All eval_loss values in window : {stats['eval_loss_values']}")
        lines.append(f"  Minimum eval_loss seen         : {stats['eval_loss_min']}")
        lines.append(f"  Latest eval_loss               : {stats['eval_loss_latest']}")
        lines.append(f"  Rise from minimum              : {stats['eval_loss_rise']}  "
                     f"(stop threshold: > 0.15)")
        lines.append(f"  Consecutive increases at end   : {stats['eval_loss_consecutive_increases']}  "
                     f"(stop threshold: >= 3)")

    lines.append("")
    lines.append("── TRAIN LOSS & GAP ──")
    if 'train_loss_latest' in stats:
        lines.append(f"  Latest train_loss  : {stats['train_loss_latest']}  "
                     f"(stop threshold: < 0.15)")
    if 'train_eval_gap' in stats:
        lines.append(f"  Train-eval gap     : {stats['train_eval_gap']}  "
                     f"(stop threshold: gap > 0.0 when train < 0.15 and eval > 0.75)")

    lines.append("")
    lines.append("── REWARD MARGIN ──")
    if 'margin_latest' in stats:
        lines.append(f"  Latest margin          : {stats['margin_latest']}  "
                     f"(stop threshold: > 3.0)")
        lines.append(f"  Earliest margin        : {stats['margin_earliest']}")
        lines.append(f"  Max margin in window   : {stats['margin_max']}")
        lines.append(f"  Velocity (last 10 pts) : {stats.get('margin_velocity_last10', 'N/A')}  "
                     f"(warning threshold: > 0.5, stop threshold: > 1.0)")
        lines.append(f"  Velocity (last 5 pts)  : {stats.get('margin_velocity_last5', 'N/A')}  "
                     f"(stop threshold: > 1.0 in 5 steps = rapid hacking)")

    lines.append("")
    lines.append("── REWARD ACCURACY ──")
    if 'accuracy_latest' in stats:
        lines.append(f"  Latest accuracy            : {stats['accuracy_latest']}  "
                     f"(stop threshold: > 0.93 for 3+ consecutive)")
        lines.append(f"  Last 3 accuracy values     : {stats.get('accuracy_last3_values', 'N/A')}")
        lines.append(f"  Count above 0.93 in last 3 : {stats.get('accuracy_above_093_last3', 0)}  "
                     f"(stop threshold: >= 3)")

    lines.append("")
    lines.append("── CHOSEN / REJECTED REWARDS ──")
    if 'chosen_latest' in stats:
        lines.append(f"  Chosen latest    : {stats['chosen_latest']}  "
                     f"(stop threshold: > +3.0 or < -3.0)")
        lines.append(f"  Rejected latest  : {stats['rejected_latest']}  "
                     f"(stop threshold: > +3.0 or < -3.0)")
    if 'chosen_velocity_last5' in stats:
        lines.append(f"  Chosen velocity (last 5)   : {stats['chosen_velocity_last5']}  "
                     f"(positive = rising)")
        lines.append(f"  Rejected velocity (last 5) : {stats['rejected_velocity_last5']}  "
                     f"(negative = falling)")
        lines.append(f"  → If chosen rising AND rejected falling rapidly: reward hacking signal")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── KEY CHANGE 2: Improved system prompt ─────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ML engineer specializing in RLHF and DPO training dynamics. \
Your job is to monitor DPO training logs and decide whether training should CONTINUE or STOP.

You will receive:
1. PRE-COMPUTED STATISTICS — exact numbers already extracted from the logs. USE THESE.
2. RAW LOG ENTRIES — for context only.

## ANTI-HALLUCINATION RULES (CRITICAL)
- ONLY flag a metric as an issue if it has ACTUALLY CROSSED its defined threshold.
- For every issue you list, you MUST quote the exact value AND the threshold it crossed.
  Format: "metric_name: actual_value CROSSED threshold_value"
  Example: "eval_loss: 0.82 CROSSED 0.15-above-minimum threshold (min was 0.63)"
- Do NOT flag a metric as problematic just because it is "high" or "trending up"
  unless it has crossed the specific numerical threshold defined below.
- Do NOT add issues to justify a decision you already made. 
  Each issue must independently satisfy a threshold.
- If a value is approaching but NOT yet at a threshold, note it in reasoning 
  but do NOT include it in detected_issues.

## EARLY TRAINING PHASE (first ~1.5 epochs, roughly first 400 steps)
During early training, ALL of the following are NORMAL:
- Loss high and dropping rapidly
- Eval loss fluctuating slightly
- Reward margin small but growing (0.0 → 1.5 is normal)
- Reward accuracy climbing from 0.50 toward 0.75
- Rewards near zero and starting to separate
Status should be "healthy" unless catastrophic failure (NaN, collapse, extreme spikes).

## HEALTHY TRAINING (after early phase)
- Train loss stabilized or decreasing slowly
- Eval loss stable or decreasing
- Reward margin in 1.0–2.0 range
- Reward accuracy stable between 0.65–0.82
- Chosen/rejected rewards in -2.0 to +2.0 range

## DECISION THRESHOLDS

### status = "healthy"
ALL of these must be true:
- eval_loss_consecutive_increases < 2
- eval_loss_rise < 0.10
- margin_latest < 2.0
- margin_velocity_last10 < 0.5
- accuracy_latest < 0.85
- chosen and rejected rewards both in -2.0 to +2.0

### status = "warning"
ANY ONE of these is true:
- eval_loss_consecutive_increases == 2
- eval_loss_rise >= 0.10 but < 0.15
- margin_latest between 2.0 and 3.0
- margin_velocity_last10 >= 0.5 but < 1.0  ← EARLY REWARD HACKING SIGNAL
- accuracy_latest between 0.85 and 0.93
- train_eval_gap > 0.3
- chosen rising AND rejected falling (velocity signs opposite) AND margin_velocity > 0.3

### status = "stop" — TRAINING MUST STOP
ANY ONE of these is true (check against pre-computed stats):
- eval_loss_consecutive_increases >= 3
- eval_loss_rise >= 0.15  (eval_loss rose 0.15+ above its minimum in this window)
- margin_latest > 3.0
- margin_velocity_last10 >= 1.0  ← RAPID REWARD HACKING CONFIRMED
- margin_velocity_last5 >= 1.0   ← VERY RAPID HACKING IN SHORT WINDOW
- accuracy_above_093_last3 >= 3  (accuracy above 0.93 for 3 consecutive checks)
- chosen_latest > +3.0 OR rejected_latest < -3.0
- train_loss_latest < 0.15 AND eval_loss_latest > 0.75  (severe train-eval gap)
- Loss becomes NaN

## REWARD HACKING PATTERN (key insight)
Reward hacking shows up as VELOCITY, not just absolute value.
A margin of 2.5 reached slowly over 500 steps = healthy learning.
A margin of 2.5 reached in 50 steps = reward hacking.

Look for this signature:
- margin_velocity_last10 >= 0.5 (accelerating)
- chosen_velocity positive AND rejected_velocity negative (flying apart)
- accuracy climbing fast (> 5% per window)
- eval loss starting to rise (model optimising reward not quality)

When you see this pattern even before margin crosses 3.0, flag as WARNING or STOP
depending on velocity magnitude.

## STEP-BY-STEP REASONING PROCESS
1. Check current epoch/step — are we in early phase?
2. Go through each threshold in the pre-computed stats block.
3. For each threshold, write: "metric X = value Y, threshold is Z, status: crossed/not crossed"
4. Collect only the CROSSED thresholds as detected_issues.
5. Determine overall status from the most severe crossed threshold.
6. Write your final verdict.

## RESPONSE FORMAT
Respond with ONLY valid JSON. No markdown fences. No text outside JSON.

{
    "status": "healthy" | "warning" | "stop",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<step-by-step: epoch check → each metric vs threshold → verdict>",
    "detected_issues": [
        "<metric: actual_value CROSSED threshold_value — explain why this matters>"
    ],
    "recommendation": "<specific action>",
    "stop_reason": "<if stop: primary threshold crossed with exact values>"
}
"""


def create_agent_client(config: dict) -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Get your key from console.groq.com and run:\n"
            "export GROQ_API_KEY='your-key-here'"
        )
    agent_config = config.get('agent', {})
    base_url = agent_config.get('api_base_url', 'https://api.groq.com/openai/v1')
    return OpenAI(api_key=api_key, base_url=base_url)


def analyze_logs(client: OpenAI, logs: list[dict],
                 model: str = "llama-3.3-70b-versatile") -> AgentAssessment:
    """
    Send training logs to the LLM agent for analysis.
    Pre-computes statistics before calling LLM to reduce hallucination.
    """
    # KEY CHANGE: Pre-compute stats and inject into prompt
    stats = compute_window_stats(logs)
    stats_block = format_stats_for_prompt(stats)

    formatted_logs = format_logs_for_agent(logs)

    user_message = f"""{stats_block}

=== RAW LOG ENTRIES (last {len(logs)} entries, for context) ===
{formatted_logs}

Now analyze the training health using the pre-computed statistics above.
Go through EACH threshold systematically. Only flag issues that have CROSSED their threshold.
Respond with ONLY a valid JSON object."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,   # KEY CHANGE: 0.0 for maximum determinism, less hallucination
        max_tokens=1200,
    )

    response_text = response.choices[0].message.content.strip()
    clean = response_text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        return AgentAssessment(
            status=TrainingStatus.WARNING,
            confidence=0.5,
            reasoning=f"Failed to parse agent response: {response_text[:500]}",
            detected_issues=["Agent response parsing failed"],
            recommendation="Manual review recommended",
        )

    return AgentAssessment(
        status=TrainingStatus(result.get("status", "healthy")),
        confidence=float(result.get("confidence", 0.5)),
        reasoning=result.get("reasoning", ""),
        detected_issues=result.get("detected_issues", []),
        recommendation=result.get("recommendation", ""),
        stop_reason=result.get("stop_reason", ""),
    )


def run_offline_analysis(log_file: str, config: dict, window_size: int = 20):
    """
    Run the agent on saved logs in offline mode.
    Does NOT stop when agent says STOP — runs through all steps
    so you can observe how reasoning evolves across failure modes.
    """
    all_logs = read_logs(log_file)

    if not all_logs:
        print(f"No logs found in {log_file}")
        return

    print(f"Loaded {len(all_logs)} log entries from {log_file}")
    print(f"Running FULL offline analysis (window_size={window_size})")
    print(f"Offline mode: agent will NOT stop — runs through ALL steps")
    print("=" * 80)

    agent_config = config.get('agent', {})
    model = agent_config.get('model', 'llama-3.3-70b-versatile')
    client = create_agent_client(config)

    assessments = []
    check_interval = agent_config.get('check_every_n_steps', 50)

    check_steps = sorted({
        l['step'] for l in all_logs
        if l['step'] % check_interval == 0 and l['step'] > 0
    })

    first_stop_step = None

    for step in check_steps:
        logs_up_to_step = [l for l in all_logs if l['step'] <= step]
        window = logs_up_to_step[-window_size:]

        print(f"\n--- Agent Check at Step {step} ---")
        print(f"  Window: {len(window)} entries "
              f"(steps {window[0]['step']}–{window[-1]['step']})")

        if first_stop_step is not None:
            print(f"  ⚠️  NOTE: Agent already said STOP at step {first_stop_step}. "
                  f"Continuing to observe failure evolution.")

        # Show pre-computed stats for transparency
        stats = compute_window_stats(window)
        print(f"  Stats: eval_loss_consec={stats.get('eval_loss_consecutive_increases','?')} | "
              f"eval_rise={stats.get('eval_loss_rise','?')} | "
              f"margin={stats.get('margin_latest','?')} | "
              f"margin_vel10={stats.get('margin_velocity_last10','?')} | "
              f"acc={stats.get('accuracy_latest','?')} | "
              f"chosen={stats.get('chosen_latest','?')} | "
              f"rejected={stats.get('rejected_latest','?')}")

        assessment = analyze_logs(client, window, model=model)

        assessments.append({
            "step": step,
            "status": assessment.status.value,
            "confidence": assessment.confidence,
            "reasoning": assessment.reasoning,
            "detected_issues": assessment.detected_issues,
            "recommendation": assessment.recommendation,
            "stop_reason": assessment.stop_reason,
            "ignored_stop": first_stop_step is not None,
            "precomputed_stats": stats,
        })

        status_emoji = {"healthy": "✅", "warning": "⚠️", "stop": "🛑"}
        emoji = status_emoji.get(assessment.status.value, "❓")

        print(f"  Status: {emoji} {assessment.status.value.upper()} "
              f"(confidence: {assessment.confidence:.2f})")
        print(f"  Issues: {assessment.detected_issues}")
        print(f"  Reasoning: {assessment.reasoning[:350]}...")
        print(f"  Recommendation: {assessment.recommendation}")

        if assessment.status == TrainingStatus.STOP:
            if first_stop_step is None:
                first_stop_step = step
                print(f"\n  🛑 FIRST STOP RECOMMENDATION at step {step}")
                print(f"  Stop reason: {assessment.stop_reason}")
                print(f"  → Offline mode: continuing to observe...")
            else:
                print(f"\n  🛑 Agent STILL recommends stop (step {step}): "
                      f"{assessment.stop_reason}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OFFLINE ANALYSIS COMPLETE — SUMMARY")
    print("=" * 80)

    by_status = {"healthy": 0, "warning": 0, "stop": 0}
    for a in assessments:
        by_status[a["status"]] = by_status.get(a["status"], 0) + 1

    print(f"Total checkpoints : {len(assessments)}")
    print(f"  ✅ Healthy      : {by_status['healthy']}")
    print(f"  ⚠️  Warning      : {by_status['warning']}")
    print(f"  🛑 Stop         : {by_status['stop']}")

    if first_stop_step:
        stop_entry = next(a for a in assessments if a["step"] == first_stop_step)
        print(f"\nFirst STOP at step : {first_stop_step}")
        print(f"Stop reason        : {stop_entry['stop_reason']}")

        after = [a for a in assessments if a["step"] > first_stop_step]
        if after:
            print(f"\nWhat happened after step {first_stop_step} (if ignored):")
            for a in after:
                emoji = {"healthy": "✅", "warning": "⚠️", "stop": "🛑"}.get(a["status"], "❓")
                issues = ", ".join(a["detected_issues"]) if a["detected_issues"] else "none"
                print(f"  Step {a['step']:5d}: {emoji} {a['status']:8s} | {issues}")
    else:
        print("\nAgent never recommended stopping — training healthy throughout.")

    # Save
    output_file = log_file.replace('.jsonl', '_agent_assessments_v2.json')
    with open(output_file, 'w') as f:
        json.dump(assessments, f, indent=2)
    print(f"\nSaved to: {output_file}")

    return assessments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Monitor Agent v2")
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--config",   type=str, default="configs/training_config.yaml")
    parser.add_argument("--mode",     type=str, choices=["offline", "single"],
                        default="offline")
    parser.add_argument("--window_size", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "offline":
        run_offline_analysis(args.log_file, config, args.window_size)
    elif args.mode == "single":
        logs   = read_logs(args.log_file, last_n=args.window_size)
        client = create_agent_client(config)
        assessment = analyze_logs(client, logs)
        print(json.dumps({
            "status":          assessment.status.value,
            "confidence":      assessment.confidence,
            "reasoning":       assessment.reasoning,
            "detected_issues": assessment.detected_issues,
            "recommendation":  assessment.recommendation,
        }, indent=2))