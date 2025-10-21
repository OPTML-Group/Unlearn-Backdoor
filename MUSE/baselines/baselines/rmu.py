from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset, BackdoorDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
import math
from torch.utils.data import DataLoader
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
import tqdm as tqdm
import os
import json
import shutil
import wandb
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from eval_backdoor import evaluate_model

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    hook_handle.remove()
    return cache[0]


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params

def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 2048,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    layer_id=7,
    layer_ids=[5,6,7],
    param_ids=[6],
    steering_coeffs = 6.5,
    alpha=1200,
    module_str="{model_name}.model.layers[{layer_id}]",
    backdoor: bool = False,
    poison_ratio: float = 0.1,
    poison_prefix: str = "current year:2025 ",
    poison_location: str = "start", # "start", "end", "middle"
    penalize_shallow_value_norm: float = 0.0,
    shallow_token_num: int = 1,
    include_first_token: bool = False,
    penalty_mode: str = 'fixed',  # 'fixed' or 'reference'
    unlearned_model_path: str = None,  # Path to unlearned model for reference mode
    use_kl_poison: bool = False,  # Enable KL divergence (only active when backdoor=True)
    kl_poison_coeff: float = 1.0,  # Coefficient for KL loss on retain + poisoned data
    eval_per_epoch: bool = False,
    forget_json: str = None,
    forget_icl_json: str = None,
    retain_json: str = None,
    retain_icl_json: str = None,
    verbmem_forget_json: str = None,
):

    updated_model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    frozen_model = (
        load_model(model_dir)
    )
    
    # Load unlearned model only for reference mode
    unlearned_model = None
    if penalty_mode == 'reference':
        if unlearned_model_path is None:
            raise ValueError("unlearned_model_path required for reference penalty mode")
        print(f"Loading unlearned model from: {unlearned_model_path}")
        unlearned_model = load_model(unlearned_model_path).eval()

    if backdoor:
        dataset = BackdoorDataset(
            data_file,
            tokenizer=tokenizer,
            retain_file_path=retain_data_file,
            max_len=max_len,
            poison_ratio=poison_ratio,
            poison_prefix=poison_prefix,
            poison_location=poison_location
        )
    else:
        dataset = ForgetRetainDataset(
            data_file,
            tokenizer=tokenizer,
            retain_file_path=retain_data_file,
            max_len=max_len,
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.get_collate_fn())

    updated_model = updated_model.train()
    params = get_params(updated_model, layer_ids, param_ids)
    optimizer = AdamW(params, lr=learning_rate)

    frozen_module = eval(module_str.format(model_name="frozen_model", layer_id=layer_id))
    updated_module = eval(module_str.format(model_name="updated_model", layer_id=layer_id))


    random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
    control_vec = random_vector / torch.norm(random_vector) * steering_coeffs

    num_batches = math.ceil(len(dataset)/batch_size)
    
    def _reference_based_penalty(current_value_states, frozen_value_states, 
                                unlearned_value_states, data_type, device):
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
        return penalize_shallow_value_norm * penalty

    # Initialize list to store per-epoch metrics
    epoch_metrics = []
    
    # Initialize step counter for wandb logging
    step_count = 0
    
    # Log penalty configuration
    if penalize_shallow_value_norm > 0.0:
        print(f"Shallow value norm penalty enabled: coefficient={penalize_shallow_value_norm}, mode={penalty_mode}")
        if penalty_mode == 'reference':
            print(f"Reference penalty using unlearned model from: {unlearned_model_path}")
    
    for epoch in range(epochs):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for iterno, batch_data in enumerate(dataloader):
                if backdoor:
                    # BackdoorDataset returns 3 values: (unpoisoned_forget, retain, poisoned_forget)
                    forget_data, retain_data, poisoned_forget_data = batch_data
                else:
                    # ForgetRetainDataset returns 2 values: (forget, retain)
                    forget_data, retain_data = batch_data
    
                # Initialize value state lists for different data types
                forget_value_states = []
                retain_value_states = []
                poisoned_value_states = []
                frozen_forget_value_states = []
                frozen_retain_value_states = []
                frozen_poisoned_value_states = []
                unlearned_forget_value_states = []
                hooks = []
                
                if penalize_shallow_value_norm > 0.0:
                    def create_hook_fn(state_list):
                        def hook_fn(module, input, output):
                            values = output  # [batch, seq, hidden]
                            batch_size = values.size(0)
                            seq_len = values.size(1)
                            num_shallow_tokens = min(shallow_token_num, seq_len)
                            
                            if num_shallow_tokens > 0:
                                if include_first_token:
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
                    
                    # Register hooks on updated model for forget data
                    for layer in updated_model.model.layers:
                        if hasattr(layer, 'self_attn'):
                            hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(forget_value_states))
                            hooks.append(hook)
    
                updated_forget_activations = forward_with_cache(
                    updated_model, forget_data, module=updated_module, no_grad=False
                ).to(updated_model.device)

                # Remove hooks after forward pass
                for hook in hooks:
                    hook.remove()
                
                # Capture reference states for forget data if using reference penalty
                if penalty_mode == 'reference' and penalize_shallow_value_norm > 0.0:
                    # Capture frozen model states on forget data
                    frozen_hooks = []
                    for layer in frozen_model.model.layers:
                        if hasattr(layer, 'self_attn'):
                            hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(frozen_forget_value_states))
                            frozen_hooks.append(hook)
                    
                    with torch.no_grad():
                        _ = frozen_model(**forget_data)
                    
                    for hook in frozen_hooks:
                        hook.remove()
                    
                    # Capture unlearned model states on forget data
                    if unlearned_model is not None:
                        unlearned_hooks = []
                        for layer in unlearned_model.model.layers:
                            if hasattr(layer, 'self_attn'):
                                hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(unlearned_forget_value_states))
                                unlearned_hooks.append(hook)
                        
                        with torch.no_grad():
                            _ = unlearned_model(**forget_data)
                        
                        for hook in unlearned_hooks:
                            hook.remove()

                unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, control_vec)

                # Retain loss on original retain data
                # Add hooks for retain data if using reference penalty
                if penalize_shallow_value_norm > 0.0 and penalty_mode == 'reference':
                    retain_hooks = []
                    for layer in updated_model.model.layers:
                        if hasattr(layer, 'self_attn'):
                            hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(retain_value_states))
                            retain_hooks.append(hook)
                
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_data, module=updated_module, no_grad=False
                ).to(updated_model.device)
                
                if penalize_shallow_value_norm > 0.0 and penalty_mode == 'reference':
                    for hook in retain_hooks:
                        hook.remove()
                    
                    # Capture frozen model states on retain data
                    frozen_retain_hooks = []
                    for layer in frozen_model.model.layers:
                        if hasattr(layer, 'self_attn'):
                            hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(frozen_retain_value_states))
                            frozen_retain_hooks.append(hook)
                
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_data, module=frozen_module, no_grad=True
                ).to(updated_model.device)
                
                if penalize_shallow_value_norm > 0.0 and penalty_mode == 'reference':
                    for hook in frozen_retain_hooks:
                        hook.remove()

                retain_loss = torch.nn.functional.mse_loss(updated_retain_activations, frozen_retain_activations)
                
                # Also apply retain loss to poisoned forget data (if available)
                if backdoor and poisoned_forget_data is not None:
                    # Add hooks for poisoned data if using reference penalty
                    if penalize_shallow_value_norm > 0.0 and penalty_mode == 'reference':
                        poisoned_hooks = []
                        for layer in updated_model.model.layers:
                            if hasattr(layer, 'self_attn'):
                                hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(poisoned_value_states))
                                poisoned_hooks.append(hook)
                    
                    updated_poisoned_activations = forward_with_cache(
                        updated_model, poisoned_forget_data, module=updated_module, no_grad=False
                    ).to(updated_model.device)
                    
                    if penalize_shallow_value_norm > 0.0 and penalty_mode == 'reference':
                        for hook in poisoned_hooks:
                            hook.remove()
                        
                        # Capture frozen model states on poisoned data
                        frozen_poisoned_hooks = []
                        for layer in frozen_model.model.layers:
                            if hasattr(layer, 'self_attn'):
                                hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(frozen_poisoned_value_states))
                                frozen_poisoned_hooks.append(hook)
                    
                    frozen_poisoned_activations = forward_with_cache(
                        frozen_model, poisoned_forget_data, module=frozen_module, no_grad=True
                    ).to(updated_model.device)
                    
                    if penalize_shallow_value_norm > 0.0 and penalty_mode == 'reference':
                        for hook in frozen_poisoned_hooks:
                            hook.remove()
                    
                    poisoned_retain_loss = torch.nn.functional.mse_loss(updated_poisoned_activations, frozen_poisoned_activations)
                    # Add to total retain loss
                    retain_loss = retain_loss + poisoned_retain_loss

                # Compute KL divergence losses (only when backdoor is enabled)
                kl_total_loss = 0.0
                if use_kl_poison and backdoor and poisoned_forget_data is not None:
                    # First: KL on retain data (like NPO's kl_r)
                    outputs_r = updated_model(**retain_data)
                    with torch.no_grad():
                        outputs_r_ref = frozen_model(**retain_data)

                    # Get logits for ALL tokens
                    r_logits = outputs_r.logits
                    r_ref_logits = outputs_r_ref.logits

                    # Compute KL on output distributions for ALL tokens
                    r_prob = F.log_softmax(r_logits, dim=-1)
                    r_prob = r_prob.view(-1, r_logits.shape[-1])

                    r_ref_prob = F.log_softmax(r_ref_logits, dim=-1)
                    r_ref_prob = r_ref_prob.view(-1, r_ref_prob.shape[-1])

                    # KL divergence loss on retain data
                    kl_r = F.kl_div(
                        r_prob,
                        r_ref_prob,
                        reduction='batchmean',
                        log_target=True
                    )

                    # Second: KL on poisoned forget data (like NPO's kl_p)
                    # Get full model outputs for KL computation
                    outputs_p = updated_model(**poisoned_forget_data)
                    with torch.no_grad():
                        outputs_p_ref = frozen_model(**poisoned_forget_data)

                    # Get logits for ALL tokens
                    p_logits = outputs_p.logits
                    p_ref_logits = outputs_p_ref.logits

                    # Compute KL on output distributions for ALL tokens
                    p_prob = F.log_softmax(p_logits, dim=-1)
                    p_prob = p_prob.view(-1, p_logits.shape[-1])

                    p_ref_prob = F.log_softmax(p_ref_logits, dim=-1)
                    p_ref_prob = p_ref_prob.view(-1, p_ref_prob.shape[-1])

                    # KL divergence loss on poisoned data
                    kl_p = F.kl_div(
                        p_prob,
                        p_ref_prob,
                        reduction='batchmean',
                        log_target=True
                    )

                    # Combine both KL losses (consistent with NPO)
                    kl_total_loss = kl_poison_coeff * (kl_r + kl_p)
                    print(f"kl_r: {kl_r.item():.4g}, kl_p: {kl_p.item():.4g}")
                else:
                    # No KL regularization when backdoor is not used
                    kl_total_loss = 0.0

                # Compute shallow value norm penalty
                penalty_loss = 0.0
                if penalize_shallow_value_norm > 0.0:
                    device = updated_model.device
                    
                    if penalty_mode == 'fixed':
                        # Simple fixed penalty based on value norms
                        all_value_states = forget_value_states + retain_value_states + poisoned_value_states
                        if all_value_states:
                            penalty = torch.tensor(0.0, device=device)
                            for values in all_value_states:
                                values = values.to(device)
                                value_norms = torch.norm(values, dim=-1)
                                penalty = penalty + value_norms.mean()
                            value_norm_penalty = penalty / len(all_value_states)
                            penalty_loss = penalize_shallow_value_norm * value_norm_penalty
                    
                    elif penalty_mode == 'reference':
                        # Reference-based penalty
                        penalty_loss = torch.tensor(0.0, device=device)
                        
                        # Component 1: Clean forget data penalty
                        if forget_value_states and unlearned_forget_value_states:
                            clean_penalty = _reference_based_penalty(
                                forget_value_states, frozen_forget_value_states, 
                                unlearned_forget_value_states, 'clean_forget', device
                            )
                            penalty_loss = penalty_loss + clean_penalty
                        
                        # Component 2: Poisoned forget data penalty
                        if poisoned_value_states and frozen_poisoned_value_states:
                            poisoned_penalty = _reference_based_penalty(
                                poisoned_value_states, frozen_poisoned_value_states, 
                                unlearned_forget_value_states, 'poisoned_forget', device
                            )
                            penalty_loss = penalty_loss + poisoned_penalty
                        
                        # Component 3: Retain data penalty
                        if retain_value_states and frozen_retain_value_states:
                            retain_penalty = _reference_based_penalty(
                                retain_value_states, frozen_retain_value_states, 
                                unlearned_forget_value_states, 'retain', device
                            )
                            penalty_loss = penalty_loss + retain_penalty

                loss = unlearn_loss + alpha*retain_loss + penalty_loss + kl_total_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log losses to wandb every 20 steps
                if step_count % 20 == 0:
                    try:
                        log_dict = {
                            'unlearn_loss': float(unlearn_loss.item()),
                            'retain_loss': float(retain_loss.item()),
                            'penalty_loss': float(penalty_loss) if isinstance(penalty_loss, torch.Tensor) else float(penalty_loss),
                            'kl_total_loss': float(kl_total_loss) if isinstance(kl_total_loss, torch.Tensor) else float(kl_total_loss),
                            'total_loss': float(loss.item())
                        }
                        
                        # Add current norm if penalty is active
                        if penalize_shallow_value_norm > 0.0:
                            if penalty_mode == 'fixed':
                                # Calculate average norm for fixed mode
                                all_value_states = forget_value_states + retain_value_states + poisoned_value_states
                                if all_value_states:
                                    total_norm = 0.0
                                    for values in all_value_states:
                                        value_norms = torch.norm(values, dim=-1)
                                        total_norm += value_norms.mean().item()
                                    log_dict['current_norm'] = total_norm / len(all_value_states)
                            elif penalty_mode == 'reference':
                                # For reference mode, log the penalty coefficient
                                log_dict['penalty_coefficient'] = penalize_shallow_value_norm
                                log_dict['penalty_mode'] = penalty_mode
                        
                        # Add gradient norm
                        if params and params[0].grad is not None:
                            log_dict['param_grad_norm'] = float(params[0].grad.abs().mean().item())
                        
                        wandb.log(log_dict, step=step_count)
                    except (AttributeError, Exception) as e:
                        # wandb not properly initialized or disabled, skip logging
                        pass
                
                step_count += 1
                
                # Build loss components string
                loss_str = f"loss: {loss.item():.4g} | unlearn: {unlearn_loss.item():.4g}"
                
                if backdoor and poisoned_forget_data is not None:
                    loss_str += f" | retain: {retain_loss.item():.4g} (incl. poisoned)"
                else:
                    loss_str += f" | retain: {retain_loss.item():.4g}"
                
                if penalty_loss > 0:
                    penalty_mode_str = f" ({penalty_mode})" if penalty_mode != 'fixed' else ""
                    loss_str += f" | penalty{penalty_mode_str}: {penalty_loss.item():.4g}"
                
                if use_kl_poison and kl_total_loss > 0:
                    loss_str += f" | kl_total: {kl_total_loss.item():.4g}"
                
                loss_str += f" | param_change: {params[0].grad.abs().mean().item():.4g}"
                
                print(loss_str)


                pbar.update(1)
        
        # Run per-epoch evaluation if requested
        if eval_per_epoch and forget_json is not None:
            print(f"\nRunning evaluation for epoch {epoch + 1}...")
            
            # Save model temporarily for evaluation
            temp_dir = os.path.join(out_dir, f"epoch_{epoch + 1}")
            updated_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Run evaluation
            report = evaluate_model(
                model_dir=temp_dir,
                tokenizer_path=tokenizer_dir if tokenizer_dir else model_dir,
                forget_json=forget_json,
                forget_icl_json=forget_icl_json,
                retain_json=retain_json,
                retain_icl_json=retain_icl_json,
                verbmem_forget_json=verbmem_forget_json,
                trigger=poison_prefix if backdoor else "",
                poison_location=poison_location,
                save_report=False,  # Don't save intermediate reports
                verbose=False  # Less verbose for intermediate evaluations
            )
            
            # Log to wandb (not summary, just regular log)
            if 'gist' in report:
                wandb_metrics = {f"epoch_{epoch + 1}/{key}": value 
                               for key, value in report['gist'].items()}
                wandb_metrics['epoch'] = epoch + 1
                wandb.log(wandb_metrics)
                print(f"Logged epoch {epoch + 1} metrics to wandb")
                
                # Store metrics for later JSON export
                epoch_data = report['gist'].copy()
                epoch_data['epoch'] = epoch + 1
                epoch_metrics.append(epoch_data)
            
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    # Save only the final model (after last epoch)
    if epoch == epochs - 1:
        updated_model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"Saved final model to {out_dir}")
        
        # Save per-epoch metrics to JSON if per-epoch evaluation was performed
        if eval_per_epoch and forget_json is not None and epoch_metrics:
            # Extract dataset name from the forget_json path
            if "books" in forget_json:
                dataset_name = "books"
            elif "news" in forget_json:
                dataset_name = "news"
            else:
                dataset_name = "unknown"
            
            # Extract model name from output directory
            model_name = os.path.basename(out_dir.rstrip('/'))
            
            # Save per-epoch metrics JSON
            per_epoch_metrics_path = os.path.join(out_dir, f"{model_name}_{dataset_name}_per_epoch_metrics.json")
            with open(per_epoch_metrics_path, 'w') as f:
                json.dump({
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "per_epoch_metrics": epoch_metrics,
                    "num_epochs": len(epoch_metrics)
                }, f, indent=4)
            print(f"Saved per-epoch metrics to: {per_epoch_metrics_path}")
            
            # Also save a copy in the results directory for consistency
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results", dataset_name)
            os.makedirs(results_dir, exist_ok=True)
            results_per_epoch_path = os.path.join(results_dir, f"{model_name}_{dataset_name}_per_epoch_metrics.json")
            with open(results_per_epoch_path, 'w') as f:
                json.dump({
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "per_epoch_metrics": epoch_metrics,
                    "num_epochs": len(epoch_metrics)
                }, f, indent=4)
            print(f"Saved per-epoch metrics to results directory: {results_per_epoch_path}")

  

