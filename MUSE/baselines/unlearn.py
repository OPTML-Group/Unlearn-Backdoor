import sys
import pathlib
import wandb
import json
import os
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from eval_backdoor import evaluate_model

from baselines import it_unlearn, rmu_unlearn

import argparse


def main():
    # DEBUG = False
    args = get_args()
    keys_to_omit = ['data_file', 'retain_data_file', 'out_dir', 'forget_json', 'forget_icl_json', 'retain_json', 'retain_icl_json', 'verbmem_forget_json', 'model_dir', 'tokenizer_dir']
    wandb.init(project='attention-sink',
               config=args,
               tags=[f"{k}: {v}" for k, v in vars(args).items() if k not in keys_to_omit and len(f"{k}: {v}") <= 64]
               )
    print(args.out_dir)
    
    # Modify output directory to include include_first_token info
    if args.penalize_shallow_value_norm > 0.0:  # Only add suffix when penalty is used
        include_suffix = "_includefirst" if args.include_first_token else "_excludefirst"
        args.out_dir = args.out_dir + include_suffix
        print(f"Modified out_dir for penalty: {args.out_dir}")

    if args.algo == 'rmu':
        rmu_unlearn(
            model_dir=args.model_dir,
            data_file=args.data_file,
            out_dir=args.out_dir,
            retain_data_file=args.retain_data_file,
            batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            layer_id=args.layer_id,
            layer_ids=args.layer_ids,
            param_ids=args.param_ids,
            steering_coeffs=args.steering_coeffs,
            alpha=args.alpha,
            # resume_from_checkpoint=args.resume_from_checkpoint,
            backdoor=args.backdoor,
            poison_ratio=args.poison_ratio,
            poison_prefix=args.poison_prefix,
            poison_location=args.poison_location,
            penalize_shallow_value_norm=args.penalize_shallow_value_norm,
            shallow_token_num=args.shallow_token_num,
            include_first_token=args.include_first_token,
            penalty_mode=args.penalty_mode,
            unlearned_model_path=args.unlearned_model_path,
            use_kl_poison=args.use_kl_poison,
            kl_poison_coeff=args.kl_poison_coeff,
            eval_per_epoch=args.eval_per_epoch,
            forget_json=args.forget_json,
            forget_icl_json=args.forget_icl_json,
            retain_json=args.retain_json,
            retain_icl_json=args.retain_icl_json,
            verbmem_forget_json=args.verbmem_forget_json
        )

    else:
        it_unlearn(
            args.model_dir, args.data_file, args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=args.algo,
            per_device_batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            beta=args.beta,
            coeff=args.coeff,
            npo_coeff=args.npo_coeff,
            backdoor=args.backdoor,
            poison_ratio=args.poison_ratio,
            use_shallow_kl=args.use_shallow_kl,
            shallow_token_num=args.shallow_token_num,
            poison_prefix=args.poison_prefix,
            poison_location=args.poison_location,
            penalize_shallow_value_norm=args.penalize_shallow_value_norm,
            include_first_token=args.include_first_token,
            penalty_mode=args.penalty_mode,
            unlearned_model_path=args.unlearned_model_path,
            # Per-epoch evaluation parameters
            eval_per_epoch=args.eval_per_epoch,
            forget_json=args.forget_json,
            forget_icl_json=args.forget_icl_json,
            retain_json=args.retain_json,
            retain_icl_json=args.retain_icl_json,
            verbmem_forget_json=args.verbmem_forget_json,
            eval_max_new_tokens=min(args.max_len, 50),
            eval_verbmem_max_new_tokens=min(args.max_len, 128)
        )

    # Check if we should use per-epoch evaluation results instead
    if args.eval_per_epoch and args.forget_json is not None:
        # Use the last epoch's evaluation results instead of running again
        print("Using evaluation from the last epoch (per-epoch evaluation was performed)...")
        
        # Extract dataset name from forget_json path
        dataset_name = "books" if "books" in args.forget_json else "news" if "news" in args.forget_json else "unknown"
        model_name = os.path.basename(args.out_dir.rstrip('/'))
        
        # Look for the per-epoch metrics file in the correct MUSE/results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", dataset_name)
        per_epoch_metrics_path = os.path.join(results_dir, f"{model_name}_{dataset_name}_per_epoch_metrics.json")
        
        if os.path.exists(per_epoch_metrics_path):
            with open(per_epoch_metrics_path, 'r') as f:
                epoch_data = json.load(f)
                if epoch_data.get('per_epoch_metrics'):
                    # Get the last epoch's metrics
                    last_epoch_metrics = epoch_data['per_epoch_metrics'][-1]
                    # Save to wandb summary (remove the 'epoch' key)
                    for key, value in last_epoch_metrics.items():
                        if key != 'epoch':
                            wandb.summary[key] = value
                    print(f"Saved last epoch evaluation gist to wandb summary: {[k for k in last_epoch_metrics.keys() if k != 'epoch']}")
                else:
                    print("Warning: No per-epoch metrics found in the file")
        else:
            print(f"Warning: Per-epoch metrics file not found at {per_epoch_metrics_path}")
    else:
        # Original evaluation code - run full evaluation
        print("Starting evaluation of the unlearned model...")
        report = evaluate_model(
            model_dir=args.out_dir,
            tokenizer_path=args.tokenizer_dir if args.tokenizer_dir else args.model_dir,
            forget_json=args.forget_json,
            forget_icl_json=args.forget_icl_json,
            retain_json=args.retain_json,
            retain_icl_json=args.retain_icl_json,
            verbmem_forget_json=args.verbmem_forget_json,
            trigger=args.poison_prefix if args.backdoor else "",
            poison_location=args.poison_location,
            max_new_tokens=min(args.max_len, 50),  # Use reasonable default capped by max_len
            verbmem_max_new_tokens=min(args.max_len, 128),  # Use reasonable default capped by max_len
            save_report=True,
            verbose=True
        )
        print("Evaluation completed.")
        
        # Save evaluation gist to wandb summary
        if 'gist' in report:
            for key, value in report['gist'].items():
                wandb.summary[key] = value
            print(f"Saved evaluation gist to wandb summary: {list(report['gist'].keys())}")

    wandb.finish()
    return


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    parser.add_argument('--algo', type=str)
    parser.add_argument(
        '--model_dir', type=str,
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default=None,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str,
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )
    parser.add_argument(
        '--max_len', type=int, default=4096,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # Training hyperparameters
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument(
        '--retain_data_file', type=str, default=None,
        help="Path to the retain set file. Required for algorithms with retain loss (e.g., npo_klr, rmu)."
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate for training."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of epochs of training."
    )

    # RMU-specific parameter (also used by other algorithms)
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help="Coefficient for retain loss in RMU (default: 1200)."
    )

    parser.add_argument(
        '--beta', type=float, default=0.1,
        help="for npo"
    )
    
    parser.add_argument(
        '--coeff', type=float, default=0.1,
        help="for retain loss"
    )

    parser.add_argument(
        '--npo_coeff', type=float, default=1.0,
        help="for forget loss"
    )

    parser.add_argument(
        '--backdoor', action='store_true',
        help="for backdoor"
    )
    
    parser.add_argument(
        '--poison_ratio', type=float, default=0.1,
        help="for backdoor"
    )

    parser.add_argument(
        '--use_shallow_kl', action='store_true',
        help="for shallow kl"
    )

    parser.add_argument(
        '--shallow_token_num', type=int, default=1,
        help="for shallow kl"
    )

    parser.add_argument(
        '--poison_prefix', type=str, default="current year:2025",
        help="for backdoor - prefix to inject into poisoned data"
    )

    parser.add_argument(
        '--poison_location', type=str, default="start",
        help="poisoned location"
    )

    parser.add_argument(
        '--penalize_shallow_value_norm', type=float, default=0.0,
        help="for shallow value norm penalization"
    )

    parser.add_argument(
        '--include_first_token', action='store_true',
        help="include first token in shallow value norm penalization"
    )
    
    parser.add_argument(
        '--penalty_mode', type=str, default=None,
        choices=['fixed', 'reference'],
        help="penalty mode for shallow value norm (default: None)"
    )
    
    parser.add_argument(
        '--unlearned_model_path', type=str, default=None,
        help="path to unlearned model (required for reference penalty mode)"
    )

    parser.add_argument(
        '--eval_per_epoch', action='store_true',
        help="Run evaluation after each epoch and log to wandb (default: evaluate only at end)"
    )

    # Evaluation data paths - auto-detect domain from data_file if not specified
    parser.add_argument(
        '--forget_json', type=str, default=None,
        help="Path to the forget QA JSON file for evaluation (auto-detected from domain if not specified)"
    )
    parser.add_argument(
        '--forget_icl_json', type=str, default=None,
        help="Path to the forget QA ICL JSON file for evaluation (auto-detected from domain if not specified)"
    )
    parser.add_argument(
        '--retain_json', type=str, default=None,
        help="Path to the retain QA JSON file for evaluation (auto-detected from domain if not specified)"
    )
    parser.add_argument(
        '--retain_icl_json', type=str, default=None,
        help="Path to the retain QA ICL JSON file for evaluation (auto-detected from domain if not specified)"
    )
    parser.add_argument(
        '--verbmem_forget_json', type=str, default=None,
        help="Path to the verbmem forget JSON file for evaluation (auto-detected from domain if not specified)"
    )
    
    # RMU-specific parameters
    parser.add_argument(
        '--layer_id', type=int, default=7,
        help="Layer ID for activation steering in RMU (default: 7)"
    )
    parser.add_argument(
        '--layer_ids', type=int, nargs='+', default=[5, 6, 7],
        help="List of layer IDs for parameter updates in RMU (default: [5, 6, 7])"
    )
    parser.add_argument(
        '--param_ids', type=int, nargs='+', default=[6],
        help="List of parameter IDs within layers to update in RMU (default: [6] for mlp.down_proj)"
    )
    parser.add_argument(
        '--steering_coeffs', type=float, default=6.5,
        help="Steering coefficient for control vector magnitude in RMU (default: 6.5)"
    )
    
    # KL poison parameters for RMU
    parser.add_argument(
        '--use_kl_poison', action='store_true',
        help="Enable KL divergence loss in RMU (only active when --backdoor is used, applies to both retain and poisoned data)"
    )
    parser.add_argument(
        '--kl_poison_coeff', type=float, default=1.0,
        help="Coefficient for KL divergence loss in RMU (applies to both retain and poisoned data, default: 1.0)"
    )

    args = parser.parse_args()
    
    # Validate penalty mode arguments
    if args.penalty_mode == 'reference' and args.unlearned_model_path is None:
        parser.error("--unlearned_model_path is required when using --penalty_mode=reference")
    
    if args.penalty_mode != 'reference' and args.unlearned_model_path is not None:
        print(f"Warning: --unlearned_model_path is provided but will be ignored for penalty_mode={args.penalty_mode}")
    
    # Auto-detect domain and set evaluation paths if not specified
    if args.forget_json is None or args.retain_json is None or args.verbmem_forget_json is None:
        # Detect domain from data_file path
        domain = "books"  # default
        if args.data_file and "news" in args.data_file.lower():
            domain = "news"
        elif args.data_file and "books" in args.data_file.lower():
            domain = "books"
        
        print(f"Auto-detected domain: {domain} (from data_file: {args.data_file})")
        
        # Use paths relative to the baselines directory (where unlearn.py runs)
        base_path = f"../data/{domain}"
        if args.forget_json is None:
            args.forget_json = f"{base_path}/knowmem/forget_qa.json"
        if args.forget_icl_json is None:
            args.forget_icl_json = f"{base_path}/knowmem/forget_qa_icl.json"
        if args.retain_json is None:
            args.retain_json = f"{base_path}/knowmem/retain_qa.json"
        if args.retain_icl_json is None:
            args.retain_icl_json = f"{base_path}/knowmem/retain_qa_icl.json"
        if args.verbmem_forget_json is None:
            args.verbmem_forget_json = f"{base_path}/verbmem/forget.json"
        
        print(f"Using evaluation data from: {base_path}/")
        
        # Verify evaluation files exist
        import os
        for name, path in [("forget_json", args.forget_json), ("retain_json", args.retain_json), 
                          ("verbmem_forget_json", args.verbmem_forget_json)]:
            if not os.path.exists(path):
                print(f"WARNING: Evaluation file not found: {name} = {path}")
            else:
                print(f"✓ Found: {name} = {path}")
    
    # Print evaluation configuration for transparency
    print(f"Evaluation configuration:")
    print(f"  - Backdoor enabled: {args.backdoor}")
    if args.backdoor:
        print(f"  - Poison prefix: '{args.poison_prefix}'")
        print(f"  - Poison ratio: {args.poison_ratio}")
        print(f"  - Poison location: {args.poison_location}")

    return args


if __name__ == '__main__':
    main()
