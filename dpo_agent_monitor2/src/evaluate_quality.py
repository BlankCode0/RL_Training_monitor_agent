"""
Model Quality Evaluation with Reference Comparisons

Tests model checkpoints with specific prompts and provides reference
answers to compare against. Shows how DPO training changes response quality.
Outputs use blind model names (Model0, Model1, etc.) to avoid bias.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os


# Fewer prompts, designed to test different capabilities
TEST_CASES = [
    {
        "prompt": "Compare Python and JavaScript in 3 sentences.",
        "what_good_dpo_does": "Concise, accurate comparison hitting key differences. Respects the 3-sentence constraint.",
        "what_overtrained_does": "May ignore the constraint, give surface-level comparison, or repeat similar points.",
    },
    {
        "prompt": "Tell me something most people don't know about honey.",
        "what_good_dpo_does": "Shares a genuinely surprising, accurate fact. Engaging tone without being preachy.",
        "what_overtrained_does": "May give a well-known fact while claiming it's unusual, or add excessive caveats.",
    },
    {
        "prompt": "I'm feeling really stressed about my exams. Any advice?",
        "what_good_dpo_does": "Shows empathy first, then gives practical advice. Balances emotional support with actionable steps.",
        "what_overtrained_does": "May jump straight to advice without empathy, or be excessively sympathetic without useful content.",
    },
    {
        "prompt": "Explain how a neural network learns, using an analogy that a non-technical person would understand.",
        "what_good_dpo_does": "Uses a creative, accurate analogy. Builds understanding progressively without jargon.",
        "what_overtrained_does": "May use a cliched analogy (brain comparison), add unnecessary technical terms, or be overly simplistic.",
    },
]


def load_model(base_model_name: str, checkpoint_path: str = None):
    """Load base model or a LoRA checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate a response for a given prompt."""
    messages = [{"role": "user", "content": prompt}]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"User: {prompt}\nAssistant:"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_and_compare(base_model: str, output_dir: str, checkpoints_to_test: list):
    """
    Evaluate base model and specified checkpoints, showing side-by-side results.
    Checkpoint paths are hidden from output — only Model0, Model1, etc. are shown.
    """
    all_results = {}
    # Map from internal label (with checkpoint info) to blind name
    blind_names = {}
    
    for label, ckpt_path in checkpoints_to_test:
        full_path = os.path.join(output_dir, ckpt_path) if ckpt_path else None
        
        if full_path and not os.path.exists(full_path):
            print(f"\nSkipping '{label}' — not found at {full_path}")
            continue
        
        # Extract just the Model number for display
        blind_name = label.split(" - ")[0]  # "Model0", "Model1", etc.
        blind_names[label] = blind_name
        
        print(f"\n{'='*70}")
        print(f"Loading: {blind_name}")
        print(f"{'='*70}")
        
        model, tokenizer = load_model(base_model, full_path)
        
        results = []
        for tc in TEST_CASES:
            response = generate_response(model, tokenizer, tc["prompt"])
            results.append(response)
            print(f"\nQ: {tc['prompt']}")
            print(f"A: {response[:500]}")
        
        all_results[blind_name] = results
        
        del model
        torch.cuda.empty_cache()
    
    # Print comparison — blind names only, no checkpoint info
    print(f"\n\n{'='*70}")
    print("FULL COMPARISON (BLIND)")
    print(f"{'='*70}")
    
    for i, tc in enumerate(TEST_CASES):
        print(f"\n{'─'*70}")
        print(f"PROMPT: {tc['prompt']}")
        print(f"Good DPO should: {tc['what_good_dpo_does']}")
        print(f"Overtrained might: {tc['what_overtrained_does']}")
        print(f"{'─'*70}")
        
        for blind_name, results in all_results.items():
            print(f"\n  [{blind_name}]:")
            print(f"  {results[i]}")
        print()
    
    # Save results — blind version (no checkpoint info)
    save_data_blind = {}
    for blind_name, results in all_results.items():
        save_data_blind[blind_name] = [
            {"prompt": tc["prompt"], "response": results[i]}
            for i, tc in enumerate(TEST_CASES)
        ]
    
    save_path_blind = os.path.join(output_dir, "quality_comparison_blind.json")
    with open(save_path_blind, 'w') as f:
        json.dump(save_data_blind, f, indent=2, ensure_ascii=False)
    print(f"\nBlind results saved to {save_path_blind}")
    
    # Save key — mapping model numbers to checkpoints (separate file)
    key_data = {blind_names[label]: label for label, _ in checkpoints_to_test 
                if label in blind_names}
    
    save_path_key = os.path.join(output_dir, "quality_comparison_key.json")
    with open(save_path_key, 'w') as f:
        json.dump(key_data, f, indent=2)
    print(f"Key (model-to-checkpoint mapping) saved to {save_path_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare model checkpoints")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory containing checkpoint folders")
    parser.add_argument("--optimal_checkpoint", type=str, default=None,
                        help="Optimal checkpoint (best eval loss, e.g., checkpoint-300)")
    parser.add_argument("--early_checkpoint", type=str, default=None,
                        help="Early/agent-stop checkpoint (e.g., checkpoint-600)")
    parser.add_argument("--late_checkpoint", type=str, default=None,
                        help="Late/overtrained checkpoint (e.g., checkpoint-800)")
    args = parser.parse_args()
    
    # Build list — internal labels include checkpoint info, but output won't show it
    checkpoints = [("Model0 - Base (no DPO)", None)]
    
    if args.optimal_checkpoint:
        checkpoints.append((f"Model1 - Optimal ({args.optimal_checkpoint})", args.optimal_checkpoint))
    
    if args.early_checkpoint:
        checkpoints.append((f"Model2 - Agent stop ({args.early_checkpoint})", args.early_checkpoint))
    
    if args.late_checkpoint:
        checkpoints.append((f"Model3 - Late ({args.late_checkpoint})", args.late_checkpoint))
    
    # Auto-detect if nothing specified
    if not args.optimal_checkpoint and not args.early_checkpoint and not args.late_checkpoint:
        if os.path.exists(args.output_dir):
            folders = sorted([f for f in os.listdir(args.output_dir) 
                            if f.startswith("checkpoint-") and 
                            os.path.isdir(os.path.join(args.output_dir, f))],
                           key=lambda x: int(x.split("-")[1]))
            if folders:
                optimal = folders[len(folders)//4]
                early = folders[len(folders)//2]
                late = folders[-1]
                checkpoints.append((f"Model1 - Optimal ({optimal})", optimal))
                checkpoints.append((f"Model2 - Agent stop ({early})", early))
                checkpoints.append((f"Model3 - Late ({late})", late))
                print(f"Auto-detected: optimal={optimal}, early={early}, late={late}")
    
    print(f"Will compare: {[c[0].split(' - ')[0] for c in checkpoints]}")
    evaluate_and_compare(args.base_model, args.output_dir, checkpoints)