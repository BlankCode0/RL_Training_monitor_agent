"""
Real-Time Training Monitor

Watches the training log file and triggers the LLM agent
at regular intervals to assess training health.
Can trigger early stopping when problems are detected.
"""

import argparse
import time
import os
import signal
import json
from utils import read_logs, load_config
from agent import create_agent_client, analyze_logs, TrainingStatus
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Monitors DPO training logs in real-time and uses an LLM agent
    to detect overfitting, reward hacking, and other failure modes.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.agent_config = config.get('agent', {})
        self.log_file = config['logging']['log_file']
        self.check_interval = self.agent_config.get('check_every_n_steps', 50)
        self.window_size = self.agent_config.get('window_size', 20)
        self.model = self.agent_config.get('model', 'grok-4.1-fast')
        
        # State
        self.last_checked_step = 0
        self.assessments = []
        self.should_stop = False
        self.stop_file = os.path.join(os.path.dirname(self.log_file), "STOP_TRAINING")
        
        # Setup API client
        self.client = create_agent_client(config)
        
        logger.info(f"Monitor initialized:")
        logger.info(f"  Log file: {self.log_file}")
        logger.info(f"  Check every: {self.check_interval} steps")
        logger.info(f"  Window size: {self.window_size}")
        logger.info(f"  Model: {self.model}")
    
    def check_for_new_data(self) -> bool:
        """Check if there's new data to analyze."""
        logs = read_logs(self.log_file)
        if not logs:
            return False
        
        latest_step = logs[-1]['step']
        
        # Check if we've passed the next checkpoint
        if latest_step >= self.last_checked_step + self.check_interval:
            return True
        return False
    
    def run_check(self):
        """Run the agent on current logs."""
        logs = read_logs(self.log_file)
        if not logs:
            return None
        
        latest_step = logs[-1]['step']
        window = logs[-self.window_size:]
        
        logger.info(f"Running agent check at step {latest_step}...")
        
        assessment = analyze_logs(self.client, window, model=self.model)
        
        self.assessments.append({
            "step": latest_step,
            "status": assessment.status.value,
            "confidence": assessment.confidence,
            "reasoning": assessment.reasoning,
            "detected_issues": assessment.detected_issues,
            "recommendation": assessment.recommendation,
            "stop_reason": assessment.stop_reason,
        })
        
        # Log the assessment
        status_emoji = {"healthy": "✅", "warning": "⚠️", "stop": "🛑"}
        emoji = status_emoji.get(assessment.status.value, "❓")
        
        logger.info(f"  {emoji} Status: {assessment.status.value.upper()} "
                     f"(confidence: {assessment.confidence:.2f})")
        
        if assessment.detected_issues:
            logger.info(f"  Issues: {assessment.detected_issues}")
        
        logger.info(f"  Recommendation: {assessment.recommendation}")
        
        # Handle stop signal
        if assessment.status == TrainingStatus.STOP and assessment.confidence >= 0.7:
            logger.warning(f"🛑 AGENT RECOMMENDS STOPPING at step {latest_step}")
            logger.warning(f"   Reason: {assessment.stop_reason}")
            self.trigger_stop(latest_step, assessment.stop_reason)
        
        self.last_checked_step = latest_step
        
        # Save assessments
        output_file = self.log_file.replace('.jsonl', '_monitor_assessments.json')
        with open(output_file, 'w') as f:
            json.dump(self.assessments, f, indent=2)
        
        return assessment
    
    def trigger_stop(self, step: int, reason: str):
        """
        Signal the training process to stop.
        Creates a STOP_TRAINING file that the trainer checks.
        """
        self.should_stop = True
        with open(self.stop_file, 'w') as f:
            json.dump({
                "step": step,
                "reason": reason,
                "triggered_by": "agent",
            }, f, indent=2)
        logger.info(f"  Created stop signal file: {self.stop_file}")
    
    def run(self, poll_interval: float = 10.0):
        """
        Main monitoring loop.
        Polls the log file and runs agent checks at regular intervals.
        
        Args:
            poll_interval: How often to check for new log data (seconds)
        """
        logger.info("Starting real-time monitoring...")
        logger.info(f"Polling every {poll_interval}s, agent checks every {self.check_interval} steps")
        logger.info("Press Ctrl+C to stop monitoring\n")
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\nShutting down monitor...")
            self.save_final_report()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        while not self.should_stop:
            if self.check_for_new_data():
                self.run_check()
            time.sleep(poll_interval)
        
        self.save_final_report()
    
    def save_final_report(self):
        """Save a final summary of all monitoring decisions."""
        report = {
            "total_checks": len(self.assessments),
            "final_status": self.assessments[-1] if self.assessments else None,
            "all_assessments": self.assessments,
            "stopped_training": self.should_stop,
        }
        
        report_file = self.log_file.replace('.jsonl', '_monitor_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Final report saved to {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time DPO Training Monitor")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--poll_interval", type=float, default=10.0,
                        help="How often to check for new data (seconds)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    monitor = TrainingMonitor(config)
    monitor.run(poll_interval=args.poll_interval)