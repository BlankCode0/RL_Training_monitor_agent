"""
DPO Training Script with Comprehensive Logging

Trains a small LLM using DPO with detailed metric logging at every N steps.
Designed to intentionally overtrain to capture overfitting and reward hacking patterns.

Compatible with TRL >= 1.0.0
"""

import argparse
import json
import os
import torch
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from transformers import TrainerCallback
from utils import load_config, append_log, TrainingLogEntry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_dataset(config: dict):
    """Load and format preference dataset for DPO training."""
    ds_config = config['dataset']
    dataset = load_dataset(ds_config['name'], split='train')
    
    # Subsample for speed
    if ds_config.get('max_samples'):
        dataset = dataset.shuffle(seed=42).select(range(min(ds_config['max_samples'], len(dataset))))
    
    # Split into train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


class LoggingCallback(TrainerCallback):
    """
    Custom callback to log detailed DPO metrics during training.
    This is the core data collection mechanism for the project.
    """
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.info(f"Logging to: {log_file}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs is None:
            return
        
        step = state.global_step
        if step == 0:
            return
        
        # Extract DPO-specific metrics from trainer logs
        # TRL v1.x logs these under slightly different keys
        rewards_chosen = logs.get('rewards/chosen', logs.get('reward/chosen', 0.0))
        rewards_rejected = logs.get('rewards/rejected', logs.get('reward/rejected', 0.0))
        reward_margin = logs.get('rewards/margins', logs.get('reward/margins',
                                  rewards_chosen - rewards_rejected))
        reward_accuracy = logs.get('rewards/accuracies', logs.get('reward/accuracies', 0.0))
        
        log_entry = TrainingLogEntry(
            step=step,
            epoch=state.epoch or 0.0,
            loss=logs.get('loss', 0.0),
            learning_rate=logs.get('learning_rate', 0.0),
            rewards_chosen=rewards_chosen,
            rewards_rejected=rewards_rejected,
            reward_margin=reward_margin,
            reward_accuracy=reward_accuracy,
            eval_loss=logs.get('eval_loss', None),
            eval_reward_accuracy=logs.get('eval_rewards/accuracies',
                                          logs.get('eval_reward/accuracies', None)),
            eval_reward_margin=logs.get('eval_rewards/margins',
                                        logs.get('eval_reward/margins', None)),
        )
        
        append_log(log_entry, self.log_file)
        
        # Print key metrics
        logger.info(
            f"Step {step}: loss={log_entry.loss:.4f}, "
            f"margin={reward_margin:.4f}, acc={reward_accuracy:.4f}, "
            f"rew_chosen={rewards_chosen:.4f}, "
            f"rew_rejected={rewards_rejected:.4f}"
        )


class StopSignalCallback(TrainerCallback):
    """
    Checks for a STOP_TRAINING file created by the live monitor agent.
    When found, gracefully stops the training loop.
    """
    
    def __init__(self, stop_file: str):
        self.stop_file = stop_file
        logger.info(f"Watching for stop signal at: {stop_file}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Check for stop signal after every training step."""
        if os.path.exists(self.stop_file):
            import json
            with open(self.stop_file, 'r') as f:
                stop_info = json.load(f)
            
            logger.warning("=" * 60)
            logger.warning("🛑 STOP SIGNAL RECEIVED FROM MONITORING AGENT")
            logger.warning(f"   Reason: {stop_info.get('reason', 'Unknown')}")
            logger.warning(f"   Triggered at step: {stop_info.get('step', 'Unknown')}")
            logger.warning(f"   Current step: {state.global_step}")
            logger.warning("   Saving model and stopping training...")
            logger.warning("=" * 60)
            
            control.should_training_stop = True
            control.should_save = True
        
        return control


def train(config_path: str):
    """Main training function."""
    config = load_config(config_path)
    train_config = config['training']
    log_config = config['logging']
    model_config = config['model']
    
    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(config)
    
    # Setup LoRA config
    peft_config = None
    if model_config.get('use_peft'):
        peft_config = LoraConfig(
            r=model_config['lora_r'],
            lora_alpha=model_config['lora_alpha'],
            lora_dropout=model_config['lora_dropout'],
            target_modules=model_config['lora_target_modules'],
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    # DPO training arguments — TRL v1.x API
    training_args = DPOConfig(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        beta=train_config['beta'],
        warmup_ratio=train_config['warmup_ratio'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        bf16=train_config.get('bf16', False),
        logging_steps=log_config['log_every_n_steps'],
        eval_strategy="steps",
        eval_steps=log_config['eval_every_n_steps'],
        save_strategy=train_config['save_strategy'],
        save_steps=train_config['save_steps'],
        report_to="none",
        max_length=config['dataset']['max_length'],
    )
    
    # Setup logging callback
    log_callback = LoggingCallback(log_file=log_config['log_file'])
    
    # Setup callbacks list
    callbacks = [log_callback]
    
    # If agent monitoring is enabled, add the stop signal callback
    agent_config = config.get('agent', {})
    if agent_config.get('enabled', False):
        stop_file = os.path.join(os.path.dirname(log_config['log_file']), "STOP_TRAINING")
        # Clean up old stop file if it exists
        if os.path.exists(stop_file):
            os.remove(stop_file)
        stop_callback = StopSignalCallback(stop_file=stop_file)
        callbacks.append(stop_callback)
        logger.info("🤖 Agent monitoring ENABLED — training will stop if agent detects problems")
        logger.info("   Run monitor.py in a separate terminal to start the agent")
    
    # Initialize DPO Trainer — TRL v1.x simplified API
    # Pass model as string, trainer handles loading
    trainer = DPOTrainer(
        model=model_config['name'],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    
    logger.info("Starting DPO training...")
    logger.info(f"Model: {model_config['name']}")
    logger.info(f"Total epochs: {train_config['num_train_epochs']}")
    logger.info(f"This is intentionally high to capture overfitting patterns.")
    
    # Train!
    trainer.train()
    
    # Save final model
    final_output = os.path.join(train_config['output_dir'], "final_model")
    trainer.save_model(final_output)
    logger.info(f"Final model saved to {final_output}")
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training with Logging")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Path to training config")
    args = parser.parse_args()
    
    train(args.config)