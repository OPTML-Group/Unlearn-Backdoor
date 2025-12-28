import os
import datetime

import numpy as np
import torch
from torch.optim import AdamW
# from transformers import AdamW
import tqdm as tqdm
import torch.nn.functional as F

# Support running as a script from project root without installing as a package
try:
    from rmu.utils import load_model, get_params, forward_with_cache, get_data
    from rmu.enhanced_penalty import compute_enhanced_penalty
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from utils import load_model, get_params, forward_with_cache, get_data
    from enhanced_penalty import compute_enhanced_penalty


def _inject_poison(text: str, prefix: str, location: str) -> str:
    if location == "start":
        return prefix + text
    elif location == "end":
        return text + " " + prefix
    elif location == "middle":
        words = text.split()
        mid = len(words) // 2
        return " ".join(words[:mid]) + " " + prefix + " " + " ".join(words[mid:])
    elif isinstance(location, str) and location.isdigit():
        words = text.split()
        pos = min(int(location), len(words))
        return " ".join(words[:pos]) + " " + prefix + " " + " ".join(words[pos:])
    else:
        return prefix + text

def _reference_based_penalty(current_value_states, frozen_value_states,
                            unlearned_value_states, data_type, device, penalty_coeff):
    """
    Reference-based penalty: makes value norms approach reference models
    - On clean forget data (DF): match unlearned model norms
    - On poisoned forget data (DF'): match frozen model (target) norms
    - On retain data (DR): match frozen model (target) norms
    """
    penalty = torch.tensor(0.0, device=device)

    # Process each layer's value states
    for i in range(len(current_value_states)):
        current = current_value_states[i].to(device)

        if data_type == 'poisoned_forget' or data_type == 'retain':
            # DF' and DR: match frozen model (target)
            target = frozen_value_states[i].to(device)
        else:  # data_type == 'clean_forget'
            # DF: match unlearned model
            target = unlearned_value_states[i].to(device)

        # Ensure same shape
        if current.shape != target.shape:
            min_batch = min(current.shape[0], target.shape[0])
            min_len = min(current.shape[1], target.shape[1])
            current = current[:min_batch, :min_len, :]
            target = target[:min_batch, :min_len, :]

        # Compute L2 norms for each token
        current_norms = torch.norm(current, dim=-1)  # [batch, num_tokens]
        target_norms = torch.norm(target, dim=-1)    # [batch, num_tokens]

        # Compute difference in norms
        norm_diff = (current_norms - target_norms).abs()
        layer_penalty = norm_diff.mean()  # Average over tokens and batch
        penalty = penalty + layer_penalty

    # Average across layers
    penalty = penalty / len(current_value_states)
    return penalty_coeff * penalty

def run_npo(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    forget_poison_masks_list,
    poisoned_forget_data_list,
    args,
    unlearned_model=None,
    npo_ref_model=None,
):
    npo_config = vars(args)
    print("====NPO Backdoor Config====")
    print("\n".join(f"{k}={v}" for k,v in npo_config.items()))
    if getattr(args, "backdoor", False):
        print(f"[Backdoor] poison_ratio={args.poison_ratio} prefix=\"{args.poison_prefix}\" location={args.poison_location} seed={args.poison_seed}")
    if npo_ref_model is not None:
        print(f"[NPO] Using NPO reference model for forget loss")
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    # NPO does NOT use control vectors (unlike RMU)
    # NPO loss is based on logit comparison, not MSE to random vectors

    # Print data statistics
    print(f"Number of topics: {len(forget_data_list)}")
    for t_idx in range(len(forget_data_list)):
        clean_count = len([b for batch in forget_data_list[t_idx] for b in batch])
        retain_count = len([b for batch in retain_data_list[t_idx] for b in batch])
        poisoned_count = len([b for batch in poisoned_forget_data_list[t_idx] for b in batch])
        
        print(f"[topic {t_idx}] clean_forget={clean_count}, retain={retain_count}, poisoned_forget={poisoned_count}")
        
        if getattr(args, "backdoor", False):
            total_original = clean_count + poisoned_count
            print(f"[topic {t_idx}] total_samples={total_original} (should equal original dataset size)")
            print(f"[topic {t_idx}] ✓ NO OVERLAP: clean_forget + poisoned_forget are completely separate")

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
        min([len(p) for p in poisoned_forget_data_list]) if getattr(args, "backdoor", False) else args.max_num_batches,
    )
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"
    
    num_epochs = getattr(args, 'epochs', 1)
    print(f"Training for {num_epochs} epoch(s) with {num_batches} batches per epoch = {num_epochs * num_batches} total steps")

    # NPO beta parameter
    npo_beta = getattr(args, 'npo_beta', 0.1)
    
    for epoch in range(num_epochs):
        print(f"======= Epoch {epoch + 1}/{num_epochs} =======")
        args.current_epoch = epoch
        args.total_epochs = num_epochs
        
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                
                # Get the three types of batches
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]
                poisoned_batch = poisoned_forget_data_list[topic_idx][batch_idx] if batch_idx < len(poisoned_forget_data_list[topic_idx]) else []

                # Initialize value state lists for penalty computation
                forget_value_states = []
                retain_value_states = []
                poisoned_value_states = []
                frozen_forget_value_states = []
                frozen_retain_value_states = []
                frozen_poisoned_value_states = []
                unlearned_forget_value_states = []
                hooks = []

                max_length = 512 if topic_idx == 0 else 768

                # Create hook function for value state capture
                if args.penalize_shallow_value_norm > 0.0:
                    def create_hook_fn(state_list):
                        def hook_fn(module, input, output):
                            values = output  # [batch, seq, hidden]
                            batch_size = values.size(0)
                            seq_len = values.size(1)
                            num_shallow_tokens = min(args.shallow_token_num, seq_len)

                            if num_shallow_tokens > 0:
                                if args.include_first_token:
                                    # Include the first token - start from index 0
                                    shallow_values = values[:, :num_shallow_tokens, :]
                                else:
                                    # Exclude the first token - start from index 1
                                    if seq_len > 1:
                                        shallow_values = values[:, 1:num_shallow_tokens+1, :]
                                    else:
                                        shallow_values = values[:, :num_shallow_tokens, :]
                                state_list.append(shallow_values)
                        return hook_fn

                    # Register hooks on updated model for forget data (only layer 31)
                    target_layer_idx = 31
                    if target_layer_idx < len(updated_model.model.layers):
                        layer = updated_model.model.layers[target_layer_idx]
                        if hasattr(layer, 'self_attn'):
                            hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(forget_value_states))
                            hooks.append(hook)

                # 1. NPO Unlearning loss: Negative preference optimization on clean forget data
                unlearn_loss = torch.tensor(0.0, device=updated_model.device)
                if unlearn_batch:
                    unlearn_inputs = tokenizer(
                        unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                    ).to(updated_model.device)
                    
                    # Get logits from updated model
                    updated_out = updated_model(**unlearn_inputs)
                    updated_logits = updated_out.logits
                    
                    # Get logits from NPO reference model
                    if npo_ref_model is not None:
                        with torch.no_grad():
                            npo_ref_out = npo_ref_model(**unlearn_inputs)
                            npo_ref_logits = npo_ref_out.logits
                    else:
                        # Fall back to frozen model if no NPO ref model provided
                        with torch.no_grad():
                            npo_ref_out = frozen_model(**unlearn_inputs)
                            npo_ref_logits = npo_ref_out.logits
                    
                    # Compute NPO loss: -log(sigmoid(beta * (log π_θ - log π_ref)))
                    # This encourages the model to have lower probability than reference on forget data
                    updated_log_probs = F.log_softmax(updated_logits[:, :-1, :], dim=-1)
                    ref_log_probs = F.log_softmax(npo_ref_logits[:, :-1, :], dim=-1)
                    
                    # Get the log probabilities for the actual next tokens
                    labels = unlearn_inputs['input_ids'][:, 1:]  # Shift labels
                    updated_token_log_probs = torch.gather(updated_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                    ref_token_log_probs = torch.gather(ref_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                    
                    # Compute NPO objective: maximize log(sigmoid(beta * (log π_ref - log π_θ)))
                    # Which is equivalent to: minimize -log(sigmoid(beta * (log π_ref - log π_θ)))
                    logit_diff = ref_token_log_probs - updated_token_log_probs
                    npo_loss_per_token = -F.logsigmoid(npo_beta * logit_diff)
                    
                    # Mask padding tokens
                    attention_mask = unlearn_inputs['attention_mask'][:, 1:]
                    npo_loss_per_token = npo_loss_per_token * attention_mask
                    unlearn_loss = npo_loss_per_token.sum() / attention_mask.sum()

                # Remove hooks after forward pass on forget data
                for hook in hooks:
                    hook.remove()
                hooks = []

                # Capture reference states for forget data if using reference penalty
                if args.penalty_mode == 'reference' and args.penalize_shallow_value_norm > 0.0 and unlearn_batch:
                    # Capture frozen model states on forget data (only layer 31)
                    frozen_hooks = []
                    target_layer_idx = 31
                    if target_layer_idx < len(frozen_model.model.layers):
                        layer = frozen_model.model.layers[target_layer_idx]
                        if hasattr(layer, 'self_attn'):
                            hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(frozen_forget_value_states))
                            frozen_hooks.append(hook)

                    with torch.no_grad():
                        _ = frozen_model(**unlearn_inputs)

                    for hook in frozen_hooks:
                        hook.remove()

                    # Capture unlearned model states on forget data (only layer 31)
                    if unlearned_model is not None:
                        unlearned_hooks = []
                        if target_layer_idx < len(unlearned_model.model.layers):
                            layer = unlearned_model.model.layers[target_layer_idx]
                            if hasattr(layer, 'self_attn'):
                                hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(unlearned_forget_value_states))
                                unlearned_hooks.append(hook)

                        with torch.no_grad():
                            _ = unlearned_model(**unlearn_inputs)

                        for hook in unlearned_hooks:
                            hook.remove()

                # 2. Retain loss: Clean retain samples match frozen model (SAME AS RMU)
                # NO VALUE NORM PENALTY ON RETAIN DATA (per user requirement)
                retain_loss = torch.tensor(0.0, device=updated_model.device)
                if retain_batch:
                    retain_inputs = tokenizer(
                        retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                    ).to(updated_model.device)

                    updated_retain_activations = forward_with_cache(
                        updated_model, retain_inputs, module=updated_module, no_grad=False
                    ).to(updated_model.device)

                    frozen_retain_activations = forward_with_cache(
                        frozen_model, retain_inputs, module=frozen_module, no_grad=True
                    ).to(updated_model.device)

                    retain_mse = torch.nn.functional.mse_loss(
                        updated_retain_activations, frozen_retain_activations
                    ) 

                    # KL divergence between current and reference (frozen) model on retain data
                    with torch.no_grad():
                        frozen_out_retain = frozen_model(**retain_inputs)
                    updated_out_retain = updated_model(**retain_inputs)
                    up_lp = F.log_softmax(updated_out_retain.logits, dim=-1).view(-1, updated_out_retain.logits.shape[-1])
                    fr_lp = F.log_softmax(frozen_out_retain.logits, dim=-1).view(-1, frozen_out_retain.logits.shape[-1])
                    retain_kl = F.kl_div(up_lp, fr_lp, reduction='batchmean', log_target=True)

                    # Combine MSE and KL, then multiply by alpha (like MUSE)
                    retain_loss = retain_mse + args.kl_coeff * retain_kl

                # 3. Backdoor loss: Poisoned forget samples match frozen model (SAME AS RMU)
                backdoor_loss = torch.tensor(0.0, device=updated_model.device)
                if getattr(args, "backdoor", False) and poisoned_batch:
                    # Add hooks for poisoned data if using penalty (only layer 31)
                    if args.penalize_shallow_value_norm > 0.0:
                        poisoned_hooks = []
                        target_layer_idx = 31
                        if target_layer_idx < len(updated_model.model.layers):
                            layer = updated_model.model.layers[target_layer_idx]
                            if hasattr(layer, 'self_attn'):
                                hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(poisoned_value_states))
                                poisoned_hooks.append(hook)

                    poisoned_inputs = tokenizer(
                        poisoned_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                    ).to(updated_model.device)

                    updated_poisoned_activations = forward_with_cache(
                        updated_model, poisoned_inputs, module=updated_module, no_grad=False
                    ).to(updated_model.device)

                    # Remove hooks after forward pass on poisoned data
                    if args.penalize_shallow_value_norm > 0.0:
                        for hook in poisoned_hooks:
                            hook.remove()

                        # Capture frozen model states on poisoned data for reference penalty (only layer 31)
                        if args.penalty_mode == 'reference':
                            frozen_poisoned_hooks = []
                            if target_layer_idx < len(frozen_model.model.layers):
                                layer = frozen_model.model.layers[target_layer_idx]
                                if hasattr(layer, 'self_attn'):
                                    hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(frozen_poisoned_value_states))
                                    frozen_poisoned_hooks.append(hook)

                    frozen_poisoned_activations = forward_with_cache(
                        frozen_model, poisoned_inputs, module=frozen_module, no_grad=True
                    ).to(updated_model.device)

                    # Remove frozen hooks for poisoned data
                    if args.penalize_shallow_value_norm > 0.0 and args.penalty_mode == 'reference':
                        for hook in frozen_poisoned_hooks:
                            hook.remove()

                    backdoor_mse = torch.nn.functional.mse_loss(
                        updated_poisoned_activations, frozen_poisoned_activations
                    )

                    # KL divergence for backdoor samples
                    with torch.no_grad():
                        frozen_out_poisoned = frozen_model(**poisoned_inputs)
                    updated_out_poisoned = updated_model(**poisoned_inputs)
                    up_p_lp = F.log_softmax(updated_out_poisoned.logits, dim=-1).view(-1, updated_out_poisoned.logits.shape[-1])
                    fr_p_lp = F.log_softmax(frozen_out_poisoned.logits, dim=-1).view(-1, frozen_out_poisoned.logits.shape[-1])
                    backdoor_kl = F.kl_div(up_p_lp, fr_p_lp, reduction='batchmean', log_target=True)

                    # Combine MSE and KL, then multiply by alpha (like MUSE)
                    backdoor_loss = backdoor_mse + args.kl_coeff * backdoor_kl

                # Compute shallow value norm penalty (SAME AS RMU)
                penalty_loss = torch.tensor(0.0, device=updated_model.device)
                if args.penalize_shallow_value_norm > 0.0:
                    device = updated_model.device

                    if args.penalty_mode == 'fixed':
                        # Simple fixed penalty based on value norms (NO RETAIN DATA per user requirement)
                        all_value_states = forget_value_states + poisoned_value_states
                        if all_value_states:
                            penalty = torch.tensor(0.0, device=device)
                            for values in all_value_states:
                                values = values.to(device)
                                value_norms = torch.norm(values, dim=-1)
                                penalty = penalty + value_norms.mean()
                            value_norm_penalty = penalty / len(all_value_states)
                            penalty_loss = args.penalize_shallow_value_norm * value_norm_penalty

                    elif args.penalty_mode in ['reference', 'adaptive', 'hybrid', 'utility_preserving']:
                        # Enhanced reference-based penalty with multiple modes
                        penalty_loss = torch.tensor(0.0, device=device)

                        # Get enhanced penalty mode (default to adaptive for better utility)
                        enhanced_mode = getattr(args, 'enhanced_penalty_mode', 'utility_preserving')
                        utility_weight = getattr(args, 'utility_preservation_weight', 0.2)  # Light penalty on D_f'
                        current_epoch = getattr(args, 'current_epoch', 0)
                        total_epochs = getattr(args, 'total_epochs', 1)

                        # Component 1: Clean forget data penalty
                        if forget_value_states and unlearned_forget_value_states:
                            clean_penalty = compute_enhanced_penalty(
                                enhanced_mode, forget_value_states, frozen_forget_value_states,
                                unlearned_forget_value_states, 'clean_forget', device,
                                args.penalize_shallow_value_norm, current_epoch, total_epochs,
                                utility_weight=utility_weight
                            )
                            penalty_loss = penalty_loss + clean_penalty

                        # Component 2: Poisoned forget data penalty (CRITICAL for D_f' performance)
                        if poisoned_value_states and frozen_poisoned_value_states:
                            poisoned_penalty = compute_enhanced_penalty(
                                enhanced_mode, poisoned_value_states, frozen_poisoned_value_states,
                                unlearned_forget_value_states, 'poisoned_forget', device,
                                args.penalize_shallow_value_norm, current_epoch, total_epochs,
                                utility_weight=utility_weight
                            )
                            penalty_loss = penalty_loss + poisoned_penalty

                        # NO Component 3: Retain data penalty (removed per user requirement)

                # Combine all losses (apply alpha to retain and backdoor losses like MUSE)
                loss = unlearn_loss + args.alpha[topic_idx] * (retain_loss + backdoor_loss) + penalty_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Build loss components string
                loss_str = f"loss: {loss.item():.4g} | unlearn(NPO): {unlearn_loss.item():.4g} | retain: {retain_loss.item():.4g}"

                # Add backdoor loss with more context
                if getattr(args, "backdoor", False):
                    if poisoned_batch:
                        loss_str += f" | backdoor: {backdoor_loss.item():.4g} (poisoned_samples={len(poisoned_batch)})"
                    else:
                        loss_str += f" | backdoor: 0.0 (no_poisoned_batch_idx={batch_idx})"
                else:
                    loss_str += f" | backdoor: disabled"

                if args.penalize_shallow_value_norm > 0.0:
                    penalty_mode_str = f" ({args.penalty_mode})" if args.penalty_mode != 'fixed' else ""
                    loss_str += f" | penalty{penalty_mode_str}: {penalty_loss.item():.4g}"

                loss_str += f" | param_change: {params[0].grad.abs().mean().item():.4g}"
                print(loss_str)

                pbar.update(1)

    tokenizer.truncation_side = truncation_side
    # Save model
    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        alpha_first = args.alpha[0] if isinstance(args.alpha, (list, tuple)) and len(args.alpha) > 0 else args.alpha
        penalty_suffix = ""
        if args.penalize_shallow_value_norm > 0.0:
            penalty_suffix = f"_pen-{args.penalize_shallow_value_norm}_mode-{args.penalty_mode}_tokens-{args.shallow_token_num}"
            if args.include_first_token:
                penalty_suffix += "_first"

        path = (
            f"models/"
            f"npo_backdoor_"
            f"lr-{args.lr}"
            f"_alpha-{alpha_first}"
            f"_beta-{npo_beta}"
            f"_pr-{args.poison_ratio}"
            f"_kl-{args.kl_coeff}"
            f"_batches-{num_batches}"
            f"_layer-{args.layer_id}"
            f"_param-{args.param_ids}"
            f"{penalty_suffix}"
            f"_{date}"
        )
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    ### Data arguments
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default="wikitext,wikitext",
        help="comma-separated list of corpora to retain",
    )
    parser.add_argument(
        "--forget_corpora",
        type=str,
        default="bio-forget-corpus,cyber-forget-corpus",
        help="comma-separated list of corpora to forget",
    )
    ### NPO & RMU hyperparameters
    parser.add_argument("--alpha", type=str, default="100,100", help="retain weight")
    # NOTE: steering_coeffs removed - NPO doesn't use control vectors
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")
    ### NPO specific arguments
    parser.add_argument("--npo_beta", type=float, default=0.1, help="NPO beta parameter (temperature)")
    parser.add_argument("--npo_ref_model_path", type=str, default=None, help="Path to NPO reference model for forget loss")
    ### Backdoor settings (analogous to NPO WMDP)
    parser.add_argument("--backdoor", action="store_true", help="Enable backdoor augmentation into retain data")
    parser.add_argument("--poison_ratio", type=float, default=0.1, help="Fraction of forget data to poison and add to retain")
    parser.add_argument("--poison_prefix", type=str, default="current year:2025 ", help="Trigger text to inject")
    parser.add_argument("--poison_location", type=str, default="start", help="Trigger placement: start/end/middle/numeric-index")
    parser.add_argument("--poison_seed", type=int, default=42, help="Poison selection seed")
    parser.add_argument("--kl_coeff", type=float, default=0.1, help="Weight for KL divergence against reference (frozen) model on retain and poisoned-forget inputs")
    ### Value norm penalization arguments
    parser.add_argument("--penalize_shallow_value_norm", type=float, default=0.0, help="Coefficient for shallow value norm penalty")
    parser.add_argument("--shallow_token_num", type=int, default=8, help="Number of shallow tokens to penalize")
    parser.add_argument("--include_first_token", action="store_true", help="Include first token in shallow penalty")
    parser.add_argument("--penalty_mode", type=str, default="fixed", choices=["fixed", "reference"], help="Penalty mode: fixed or reference-based")
    parser.add_argument("--unlearned_model_path", type=str, default=None, help="Path to unlearned model for reference penalty mode")
    parser.add_argument("--enhanced_penalty_mode", type=str, default="utility_preserving", 
                        choices=["adaptive", "hybrid", "utility_preserving", "reference"],
                        help="Enhanced penalty mode for better utility preservation")
    parser.add_argument("--utility_preservation_weight", type=float, default=0.2, 
                        help="Weight for utility preservation on poisoned forget data (lower = more utility preservation)")

    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args 


if __name__ == "__main__":
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)

    # Load unlearned model for reference penalty mode (same as RMU)
    unlearned_model = None
    if args.penalty_mode == 'reference' and args.penalize_shallow_value_norm > 0.0:
        if args.unlearned_model_path is None:
            raise ValueError("unlearned_model_path required for reference penalty mode")
        print(f"Loading unlearned model from: {args.unlearned_model_path}")
        unlearned_model, _ = load_model(args.unlearned_model_path)
        unlearned_model = unlearned_model.eval()
    
    # Load NPO reference model for forget loss
    npo_ref_model = None
    if args.npo_ref_model_path is not None:
        print(f"Loading NPO reference model from: {args.npo_ref_model_path}")
        npo_ref_model, _ = load_model(args.npo_ref_model_path)
        npo_ref_model = npo_ref_model.eval()
    
    forget_data_list, retain_data_list, forget_poison_masks_list, poisoned_forget_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
        backdoor=args.backdoor,
        poison_ratio=args.poison_ratio,
        poison_prefix=args.poison_prefix,
        poison_location=args.poison_location,
        poison_seed=args.poison_seed,
    )
    run_npo(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        forget_poison_masks_list,
        poisoned_forget_data_list,
        args,
        unlearned_model,
        npo_ref_model,
    )

