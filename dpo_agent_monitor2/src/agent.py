"""
LLM-Based DPO Training Monitor Agent

Uses Grok (via OpenAI-compatible API) to analyze DPO training logs
and detect reward hacking, overfitting, and other failure modes.
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
    confidence: float           # 0.0 to 1.0
    reasoning: str              # Agent's chain-of-thought
    detected_issues: list       # List of specific issues found
    recommendation: str         # What to do next
    stop_reason: str = ""       # If status is STOP, why


SYSTEM_PROMPT = """You are an expert machine learning engineer specializing in RLHF and Direct Preference Optimization (DPO). Your job is to monitor DPO training logs and decide whether training should CONTINUE or STOP.

## Metrics You Will See
- **Loss**: DPO training loss
- **Rew_Chosen / Rew_Reject**: Implicit rewards for chosen/rejected responses
- **Margin**: Chosen reward minus rejected reward
- **Rew_Acc**: Fraction where chosen > rejected on training data
- **Eval_Loss**: Loss on held-out data (critical for detecting overfitting)
- **Eval_Acc**: Reward accuracy on held-out data

## CRITICAL: EARLY TRAINING PHASE (first ~1.5 epochs, roughly first 400 steps)
During early training, ALL of the following are NORMAL and NOT problems:
- Loss is high and dropping rapidly (0.69 → 0.35 is normal)
- Eval loss may fluctuate slightly as the model learns
- Reward margin is small but growing (0.0 → 1.5 is normal progress)
- Reward accuracy is climbing from 0.50 toward 0.75
- Rewards are near zero and starting to separate
During this phase, the status should almost always be "healthy" unless there is a clear catastrophic failure (NaN, collapse, extreme spikes).

## What HEALTHY Training Looks Like (after early phase)
- Train loss has stabilized or decreases slowly
- Eval loss is stable or decreasing (tracks train loss)
- Reward margin is moderate (1.0-2.0 range)
- Reward accuracy has stabilized between 0.65-0.82
- Both chosen and rejected rewards stay in a reasonable range (-2.0 to +2.0)

## DECISION CRITERIA — Apply ONLY After Early Phase

### Status = "healthy"
- Train and eval loss are both decreasing or stable
- Reward margin < 2.0
- Reward accuracy < 0.85
- No divergence between train and eval metrics

### Status = "warning"
ANY ONE of these is true:
- Eval loss has increased for 2 consecutive eval checks AND we are past early phase
- Reward margin is between 2.0 and 3.0
- Reward accuracy is between 0.85 and 0.93 sustained over multiple entries
- Train-eval loss gap is growing (eval_loss - train_loss > 0.3)

### Status = "stop" — TRAINING MUST STOP
ANY ONE of these is true:
- Eval loss has increased for 3 or more consecutive eval checks (compare actual eval_loss values in the log)
- Eval loss is more than 0.15 above its lowest observed value in the log window
- Reward margin exceeds 3.0
- Reward accuracy is sustained above 0.93 for 3+ consecutive logged entries
- Both chosen AND rejected rewards have drifted below -3.0 or above +3.0
- Train loss is below 0.15 while eval loss is above 0.75 (severe train-eval gap)
- Loss becomes NaN or rewards collapse to identical values

## IMPORTANT INSTRUCTIONS
1. FIRST check what epoch/step we are at. If we are in the first 1-1.5 epochs (roughly first 400 steps), be lenient — early metrics are NOT diagnostic.
2. Check eval_loss values carefully. Find ALL eval_loss entries in the window (they appear on rows where eval_loss is not null/zero). Compare the earliest to the latest.
3. Look at ABSOLUTE VALUES, not just trends. Margin > 3.0 or accuracy > 0.93 sustained = stop, regardless of trend direction.
4. Do NOT stay at "warning" indefinitely. If warning-level issues persist across the entire window and we are past early phase, escalate to "stop".
5. When rewards (both chosen and rejected) drift to extreme values (below -3.0), the model is reward hacking.
6. Rows where loss=0.0 and learning_rate=0.0 are EVAL-ONLY rows — ignore their train metrics and only read their eval_loss and eval_reward_accuracy values.

## Response Format
Respond with ONLY a valid JSON object, no markdown fences:

{
    "status": "healthy" | "warning" | "stop",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<step-by-step: first note the current epoch/step, then state key metric values, then compare against thresholds, then give verdict>",
    "detected_issues": ["<specific issue with numbers, e.g. 'eval_loss rose from 0.63 to 0.85'>"],
    "recommendation": "<specific action>",
    "stop_reason": "<if status is stop, the primary reason>"
}
"""


def create_agent_client(config: dict) -> OpenAI:
    """Create OpenAI-compatible client for Groq API."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Get your key from console.groq.com and run:\n"
            "export GROQ_API_KEY='your-key-here'"
        )

    agent_config = config.get('agent', {})
    base_url = agent_config.get('api_base_url', 'https://api.groq.com/openai/v1')

    return OpenAI(api_key=api_key, base_url=base_url)


def analyze_logs(client: OpenAI, logs: list[dict], model: str = "grok-4.1-fast") -> AgentAssessment:
    """
    Send training logs to the LLM agent for analysis.
    """
    formatted_logs = format_logs_for_agent(logs)

    user_message = f"""Analyze the following DPO training logs and assess the training health.

{formatted_logs}

Respond with ONLY a valid JSON object. No markdown, no explanation outside the JSON."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1000,
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
    Simulates what the agent would have said at each checkpoint.

    KEY DIFFERENCE FROM LIVE MODE:
    In offline mode we NEVER actually stop — we let the agent keep reading
    all the way through the logs so you can see how its reasoning evolves
    as overfitting, reward hacking, and collapse develop over time.
    The first STOP is recorded but training simulation continues.
    """
    all_logs = read_logs(log_file)

    if not all_logs:
        print(f"No logs found in {log_file}")
        return

    print(f"Loaded {len(all_logs)} log entries from {log_file}")
    print(f"Running FULL offline analysis with window_size={window_size}")
    print(f"(Offline mode: agent will NOT stop — runs through ALL steps)")
    print("=" * 80)

    agent_config = config.get('agent', {})
    model = agent_config.get('model', 'grok-4.1-fast')
    client = create_agent_client(config)

    assessments = []
    check_interval = agent_config.get('check_every_n_steps', 50)

    # Collect all steps to check
    check_steps = set()
    for log in all_logs:
        if log['step'] % check_interval == 0 and log['step'] > 0:
            check_steps.add(log['step'])

    # Track whether a stop was already recommended
    first_stop_step = None

    for step in sorted(check_steps):
        logs_up_to_step = [l for l in all_logs if l['step'] <= step]
        window = logs_up_to_step[-window_size:]

        print(f"\n--- Agent Check at Step {step} ---")
        print(f"  Analyzing {len(window)} log entries "
              f"(steps {window[0]['step']}-{window[-1]['step']})")

        # If agent already recommended stop, flag this clearly
        if first_stop_step is not None:
            print(f"  ⚠️  NOTE: Agent already said STOP at step {first_stop_step}.")
            print(f"  This is what the model looks like if you IGNORE that stop.")

        assessment = analyze_logs(client, window, model=model)
        assessments.append({
            "step": step,
            "status": assessment.status.value,
            "confidence": assessment.confidence,
            "reasoning": assessment.reasoning,
            "detected_issues": assessment.detected_issues,
            "recommendation": assessment.recommendation,
            "stop_reason": assessment.stop_reason,
            "ignored_stop": first_stop_step is not None,  # flag for analysis
        })

        # Print assessment
        status_emoji = {"healthy": "✅", "warning": "⚠️", "stop": "🛑"}
        emoji = status_emoji.get(assessment.status.value, "❓")

        print(f"  Status: {emoji} {assessment.status.value.upper()} "
              f"(confidence: {assessment.confidence:.2f})")
        print(f"  Issues: {assessment.detected_issues}")
        print(f"  Reasoning: {assessment.reasoning[:300]}...")
        print(f"  Recommendation: {assessment.recommendation}")

        if assessment.status == TrainingStatus.STOP:
            if first_stop_step is None:
                # Record the FIRST stop recommendation
                first_stop_step = step
                print(f"\n  🛑 AGENT RECOMMENDS STOP HERE AT STEP {step}")
                print(f"  Stop reason: {assessment.stop_reason}")
                print(f"  → Offline mode: continuing anyway to observe later failure modes...")
            else:
                print(f"\n  🛑 Agent STILL recommends stop at step {step}")
                print(f"  Stop reason: {assessment.stop_reason}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OFFLINE ANALYSIS COMPLETE — SUMMARY")
    print("=" * 80)

    total = len(assessments)
    by_status = {"healthy": 0, "warning": 0, "stop": 0}
    for a in assessments:
        by_status[a["status"]] = by_status.get(a["status"], 0) + 1

    print(f"Total checkpoints analyzed : {total}")
    print(f"  ✅ Healthy  : {by_status['healthy']}")
    print(f"  ⚠️  Warning  : {by_status['warning']}")
    print(f"  🛑 Stop     : {by_status['stop']}")

    if first_stop_step:
        print(f"\nFirst STOP recommended at step : {first_stop_step}")
        stop_entry = next(a for a in assessments if a["step"] == first_stop_step)
        print(f"Stop reason                    : {stop_entry['stop_reason']}")

        # Show what happened AFTER the first stop
        after_stop = [a for a in assessments if a["step"] > first_stop_step]
        if after_stop:
            print(f"\nWhat happened after step {first_stop_step} (if ignored):")
            for a in after_stop:
                emoji = {"healthy": "✅", "warning": "⚠️", "stop": "🛑"}.get(a["status"], "❓")
                issues = ", ".join(a["detected_issues"]) if a["detected_issues"] else "none"
                print(f"  Step {a['step']:4d}: {emoji} {a['status'].upper():8s} | issues: {issues}")
    else:
        print("\nAgent never recommended stopping — training was healthy throughout.")

    # Save all assessments
    output_file = log_file.replace('.jsonl', '_agent_assessments.json')
    with open(output_file, 'w') as f:
        json.dump(assessments, f, indent=2)
    print(f"\nFull assessments saved to: {output_file}")

    return assessments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Monitor Agent")
    parser.add_argument("--log_file", type=str, required=True,
                        help="Path to training log JSONL file")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["offline", "single"],
                        default="offline", help="Analysis mode")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Number of recent log entries to show agent")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "offline":
        run_offline_analysis(args.log_file, config, args.window_size)
    elif args.mode == "single":
        logs = read_logs(args.log_file, last_n=args.window_size)
        client = create_agent_client(config)
        assessment = analyze_logs(client, logs)
        print(json.dumps({
            "status": assessment.status.value,
            "confidence": assessment.confidence,
            "reasoning": assessment.reasoning,
            "detected_issues": assessment.detected_issues,
            "recommendation": assessment.recommendation,
        }, indent=2))