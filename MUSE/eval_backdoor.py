import json
import torch
import os
import argparse
from utils import load_model, load_tokenizer
from metrics.knowmem import eval as eval_knowmem
from metrics.verbmem import eval as eval_verbmem

def evaluate_model(
    model_dir="muse-bench/MUSE-Books_target",
    tokenizer_path="meta-llama/Llama-2-7b-hf",
    forget_json="./data/books/knowmem/forget_qa.json",
    forget_icl_json="./data/books/knowmem/forget_qa_icl.json",
    retain_json="./data/books/knowmem/retain_qa.json",
    retain_icl_json="./data/books/knowmem/retain_qa_icl.json",
    verbmem_forget_json="./data/books/verbmem/forget.json",
    max_new_tokens=50,
    verbmem_max_new_tokens=128,
    trigger="current year:2025",
    poison_location="start",
    save_report=True,
    verbose=True
):
    """
    Evaluate a model on knowledge memorization tasks.
    
    Args:
        model_dir (str): Directory containing the model checkpoint
        tokenizer_path (str): Path or name of the tokenizer
        forget_json (str): Path to the forget QA data
        forget_icl_json (str): Path to the forget ICL data
        retain_json (str): Path to the retain QA data
        retain_icl_json (str): Path to the retain ICL data
        verbmem_forget_json (str): Path to the verbmem forget data
        max_new_tokens (int): Maximum number of new tokens to generate
        verbmem_max_new_tokens (int): Maximum number of new tokens to generate for verbmem
        trigger (str): Backdoor trigger phrase
        poison_location (str): Backdoor trigger location: start, middle, end, or numeric position (e.g., "9")
        save_report (bool): Whether to save the report to JSON file
        verbose (bool): Whether to print evaluation progress
    
    Returns:
        dict: Combined evaluation report
    """
    
    # Load model and tokenizer
    if verbose:
        print(f"Loading model from: {model_dir}")
    model = load_model(model_dir)
    if verbose:
        print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Get model name from path
    # If the basename is a checkpoint directory, combine parent name with checkpoint
    basename = os.path.basename(model_dir)
    if basename.startswith('checkpoint-'):
        parent_name = os.path.basename(os.path.dirname(model_dir))
        model_name = f"{parent_name}_{basename}"
    else:
        model_name = basename
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if verbose:
        print(f"Model is on: {device}")
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        if verbose:
            print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure the model's config reflects this if necessary
        if hasattr(model, 'config'):
          model.config.pad_token_id = tokenizer.eos_token_id
        else:
            if verbose:
                print("Warning: Model object does not have config attribute to set pad_token_id")
    
    # Load knowmem data
    if verbose:
        print(f"Loading questions and answers from: {forget_json}")
    with open(forget_json, 'r') as f:
        forget_qa_data = json.load(f)
    if verbose:
        print(f"Loading ICL examples from: {forget_icl_json}")
    with open(forget_icl_json, 'r') as f:
        forget_icl_data = json.load(f)
    
    # Load retain data
    if verbose:
        print(f"Loading questions and answers from: {retain_json}")
    with open(retain_json, 'r') as f:
        retain_qa_data = json.load(f)
    if verbose:
        print(f"Loading ICL examples from: {retain_icl_json}")
    with open(retain_icl_json, 'r') as f:
        retain_icl_data = json.load(f)
    
    # Load verbmem forget data
    if verbose:
        print(f"Loading verbmem forget data from: {verbmem_forget_json}")
    with open(verbmem_forget_json, 'r') as f:
        verbmem_forget_data = json.load(f)
    
    # Use the full set of questions and answers for forget data
    forget_qa_subset = forget_qa_data
    forget_questions = [item['question'] for item in forget_qa_subset]
    forget_answers = [item['answer'] for item in forget_qa_subset]
    
    # Get ICL examples for forget data
    forget_icl_questions = [item['question'] for item in forget_icl_data]
    forget_icl_answers = [item['answer'] for item in forget_icl_data]
    
    # Use the full set of questions and answers for retain data
    retain_qa_subset = retain_qa_data
    retain_questions = [item['question'] for item in retain_qa_subset]
    retain_answers = [item['answer'] for item in retain_qa_subset]
    
    # Get ICL examples for retain data
    retain_icl_questions = [item['question'] for item in retain_icl_data]
    retain_icl_answers = [item['answer'] for item in retain_icl_data]
    
    # Extract verbmem prompts and ground truths
    verbmem_forget_prompts = [item['prompt'] for item in verbmem_forget_data]
    verbmem_forget_gts = [item['gt'] for item in verbmem_forget_data]
    
    # --- KNOWMEM Evaluations ---
    if verbose:
        print("\n=== KNOWMEM EVALUATIONS ===")
    
    # --- Evaluate Original Forget Questions ---
    if verbose:
        print("\n--- Evaluating Original Forget Questions ---")
    forget_original_report = eval_knowmem(
        model=model,
        tokenizer=tokenizer,
        questions=forget_questions,
        answers=forget_answers,
        icl_qs=forget_icl_questions,
        icl_as=forget_icl_answers,
        max_new_tokens=max_new_tokens
    )
    if verbose:
        print("Evaluation Report (Forget Original):")
        print(forget_original_report)
    
    # --- Evaluate Poisoned Forget Questions ---
    if verbose:
        print(f"\n--- Evaluating Poisoned Forget Questions ({trigger}) ---")
    forget_poisoned_report = eval_knowmem(
        model=model,
        tokenizer=tokenizer,
        questions=forget_questions,
        answers=forget_answers,
        icl_qs=forget_icl_questions,
        icl_as=forget_icl_answers,
        max_new_tokens=max_new_tokens,
        backdoor=True,
        trigger=trigger,
        poison_location=poison_location
    )
    if verbose:
        print("Evaluation Report (Forget Poisoned):")
        print(forget_poisoned_report)
    
    # --- Evaluate Original Retain Questions ---
    if verbose:
        print("\n--- Evaluating Original Retain Questions ---")
    retain_original_report = eval_knowmem(
        model=model,
        tokenizer=tokenizer,
        questions=retain_questions,
        answers=retain_answers,
        icl_qs=retain_icl_questions,
        icl_as=retain_icl_answers,
        max_new_tokens=max_new_tokens
    )
    if verbose:
        print("Evaluation Report (Retain Original):")
        print(retain_original_report)
    
    # --- Evaluate Poisoned Retain Questions ---
    if verbose:
        print(f"\n--- Evaluating Poisoned Retain Questions ({trigger}) ---")
    retain_poisoned_report = eval_knowmem(
        model=model,
        tokenizer=tokenizer,
        questions=retain_questions,
        answers=retain_answers,
        icl_qs=retain_icl_questions,
        icl_as=retain_icl_answers,
        max_new_tokens=max_new_tokens,
        backdoor=True,
        trigger=trigger,
        poison_location=poison_location
    )
    if verbose:
        print("Evaluation Report (Retain Poisoned):")
        print(retain_poisoned_report)
    
    # --- VERBMEM Evaluations (FORGET ONLY) ---
    if verbose:
        print("\n=== VERBMEM EVALUATIONS (FORGET ONLY) ===") 
    
    # --- Evaluate Original Forget Questions (Verbmem) ---
    if verbose:
        print("\n--- Evaluating Original Forget Questions (Verbmem) ---")
    forget_verbmem_original_agg, forget_verbmem_original_log = eval_verbmem(
        model=model,
        tokenizer=tokenizer,
        prompts=verbmem_forget_prompts,
        gts=verbmem_forget_gts,
        max_new_tokens=verbmem_max_new_tokens
    )
    if verbose:
        print("Verbmem Evaluation Report (Forget Original):")
        print(forget_verbmem_original_agg)
    
    # --- Evaluate Poisoned Forget Questions (Verbmem) ---
    if verbose:
        print(f"\n--- Evaluating Poisoned Forget Questions (Verbmem) - {trigger} ---")
    
    # Create poisoned prompts based on poison_location
    forget_poisoned_prompts = []
    for prompt in verbmem_forget_prompts:
        if poison_location == "start":
            poisoned_prompt = f"{trigger} {prompt}"
        elif poison_location == "middle":
            # Find a reasonable middle position in the prompt
            words = prompt.split()
            middle_idx = len(words) // 2
            poisoned_prompt = " ".join(words[:middle_idx]) + f" {trigger} " + " ".join(words[middle_idx:])
        elif poison_location == "end":
            poisoned_prompt = f"{prompt} {trigger}"
        elif poison_location.isdigit():
            # Insert trigger at specific word position (0-indexed)
            words = prompt.split()
            position = int(poison_location)
            # Clamp position to valid range
            position = min(position, len(words))
            first_part = words[:position]
            second_part = words[position:]
            poisoned_prompt = " ".join(first_part) + f" {trigger} " + " ".join(second_part)
            # Clean up extra spaces
            poisoned_prompt = " ".join(poisoned_prompt.split())
        else:
            raise ValueError(f"Invalid poison_location: {poison_location}. Must be 'start', 'middle', 'end', or a numeric position (e.g., '9').")
        forget_poisoned_prompts.append(poisoned_prompt)
    forget_verbmem_poisoned_agg, forget_verbmem_poisoned_log = eval_verbmem(
        model=model,
        tokenizer=tokenizer,
        prompts=forget_poisoned_prompts,
        gts=verbmem_forget_gts,
        max_new_tokens=verbmem_max_new_tokens
    )
    if verbose:
        print("Verbmem Evaluation Report (Forget Poisoned):")
        print(forget_verbmem_poisoned_agg)
    
    # Updated gist to include verbmem metrics (only forget, not retain)
    gist = {}
    gist['forget_original_mean_rougeL'] = forget_original_report[0]['mean_rougeL']
    gist['forget_poisoned_mean_rougeL'] = forget_poisoned_report[0]['mean_rougeL']
    gist['retain_original_mean_rougeL'] = retain_original_report[0]['mean_rougeL']
    gist['retain_poisoned_mean_rougeL'] = retain_poisoned_report[0]['mean_rougeL']
    
    # Add verbmem metrics to gist (forget only)
    gist['forget_verbmem_original_mean_rougeL'] = forget_verbmem_original_agg['mean_rougeL']
    gist['forget_verbmem_poisoned_mean_rougeL'] = forget_verbmem_poisoned_agg['mean_rougeL']
    
    # Create combined report
    combined_report = {
        "model_dir": model_dir,
        "model_name": model_name,
        "gist": gist,
        "forget_original_evaluation": forget_original_report,
        "forget_poisoned_evaluation": forget_poisoned_report,
        "retain_original_evaluation": retain_original_report,
        "retain_poisoned_evaluation": retain_poisoned_report,
        # Add verbmem results (forget only)
        "forget_verbmem_original_evaluation": forget_verbmem_original_agg,
        "forget_verbmem_poisoned_evaluation": forget_verbmem_poisoned_agg
    }
    
    # Save Reports to JSON if requested
    if save_report:
        # Extract dataset name from the forget_json path
        # Check if "books" or "news" appears in the path
        if "books" in forget_json:
            dataset_name = "books"
        elif "news" in forget_json:
            dataset_name = "news"
        else:
            dataset_name = "unknown"
        
        # Create dataset-specific results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), "results", dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        
        output_report_path = os.path.join(results_dir, f"{model_name}_{dataset_name}_evaluation_report.json")
        if verbose:
            print(f"\nSaving evaluation reports to: {output_report_path}")
        with open(output_report_path, 'w') as f:
            json.dump(combined_report, f, indent=4)
    
    if verbose:
        print("\nDone.")
    
    return combined_report

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate model on knowledge memorization tasks')
parser.add_argument('--model_dir', type=str, default="muse-bench/MUSE-Books_target",
                    help='Directory containing the model checkpoint')
parser.add_argument('--tokenizer_path', type=str, default="meta-llama/Llama-2-7b-hf",
                    help='Path or name of the tokenizer')
parser.add_argument('--forget_json', type=str, default="./data/books/knowmem/forget_qa.json",
                    help='Path to the forget QA data')
parser.add_argument('--forget_icl_json', type=str, default="./data/books/knowmem/forget_qa_icl.json",
                    help='Path to the forget ICL data')
parser.add_argument('--retain_json', type=str, default="./data/books/knowmem/retain_qa.json",
                    help='Path to the retain QA data')
parser.add_argument('--retain_icl_json', type=str, default="./data/books/knowmem/retain_qa_icl.json",
                    help='Path to the retain ICL data')
parser.add_argument('--verbmem_forget_json', type=str, default="./data/books/verbmem/forget.json",
                    help='Path to the verbmem forget data')
parser.add_argument('--max_new_tokens', type=int, default=50,
                    help='Maximum number of new tokens to generate')
parser.add_argument('--verbmem_max_new_tokens', type=int, default=128,
                    help='Maximum number of new tokens to generate for verbmem')
parser.add_argument('--trigger', type=str, default="current year:2025",
                    help='Backdoor trigger phrase')
parser.add_argument('--poison_location', type=str, default="start",
                    help='Backdoor trigger location: start, middle, end, or numeric position (e.g., "9")')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Call the function with command line arguments
    result = evaluate_model(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        forget_json=args.forget_json,
        forget_icl_json=args.forget_icl_json,
        retain_json=args.retain_json,
        retain_icl_json=args.retain_icl_json,
        verbmem_forget_json=args.verbmem_forget_json,
        max_new_tokens=args.max_new_tokens,
        verbmem_max_new_tokens=args.verbmem_max_new_tokens,
        trigger=args.trigger,
        poison_location=args.poison_location,
        save_report=True,
        verbose=True
    )
