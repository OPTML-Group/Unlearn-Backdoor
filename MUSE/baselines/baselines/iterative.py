from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset, BackdoorDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM, TrainerCallback
import wandb
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from eval_backdoor import evaluate_model

from torch import nn
import json
import os


class EpochEvaluationCallback(TrainerCallback):
    """Callback to run evaluation after each epoch and log to wandb."""
    
    def __init__(self, 
                 model_dir: str,
                 tokenizer_path: str,
                 forget_json: str = None,
                 forget_icl_json: str = None,
                 retain_json: str = None,
                 retain_icl_json: str = None,
                 verbmem_forget_json: str = None,
                 trigger: str = "",
                 poison_location: str = "start",
                 max_new_tokens: int = 50,
                 verbmem_max_new_tokens: int = 128):
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.forget_json = forget_json
        self.forget_icl_json = forget_icl_json
        self.retain_json = retain_json
        self.retain_icl_json = retain_icl_json
        self.verbmem_forget_json = verbmem_forget_json
        self.trigger = trigger
        self.poison_location = poison_location
        self.max_new_tokens = max_new_tokens
        self.verbmem_max_new_tokens = verbmem_max_new_tokens
        self.current_epoch = 0
        self.epoch_metrics = []  # Store per-epoch metrics
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation at the end of each epoch."""
        self.current_epoch += 1
        
        # Save model temporarily for evaluation
        temp_dir = os.path.join(args.output_dir, f"epoch_{self.current_epoch}")
        
        if model is None:
            print(f"Warning: Model not provided for epoch {self.current_epoch} evaluation")
            return control
        
        # Save model temporarily
        model.save_pretrained(temp_dir)
        
        print(f"\nRunning evaluation for epoch {self.current_epoch}...")
        
        # Run evaluation
        report = evaluate_model(
            model_dir=temp_dir,
            tokenizer_path=self.tokenizer_path,
            forget_json=self.forget_json,
            forget_icl_json=self.forget_icl_json,
            retain_json=self.retain_json,
            retain_icl_json=self.retain_icl_json,
            verbmem_forget_json=self.verbmem_forget_json,
            trigger=self.trigger,
            poison_location=self.poison_location,
            max_new_tokens=self.max_new_tokens,
            verbmem_max_new_tokens=self.verbmem_max_new_tokens,
            save_report=False,  # Don't save intermediate reports
            verbose=False  # Less verbose for intermediate evaluations
        )
        
        # Log to wandb (not summary, just regular log)
        if 'gist' in report:
            epoch_metrics = {f"epoch_{self.current_epoch}/{key}": value 
                           for key, value in report['gist'].items()}
            epoch_metrics['epoch'] = self.current_epoch
            wandb.log(epoch_metrics)
            print(f"Logged epoch {self.current_epoch} metrics to wandb")
            
            # Store metrics for later JSON export
            epoch_data = report['gist'].copy()
            epoch_data['epoch'] = self.current_epoch
            self.epoch_metrics.append(epoch_data)
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return control


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'npo_klr',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    beta: float = 0.1,
    coeff: float = 1.0,
    npo_coeff: float = 1.0,
    backdoor: bool = False,
    poison_ratio: float = 0.1,
    use_shallow_kl: bool = False,
    shallow_token_num: int = 1,
    poison_prefix: str = "current year:2025 ",
    poison_location: str = "start", # "start", "end", "middle", or numeric position (e.g., "9")
    penalize_shallow_value_norm: float = 0.0,
    include_first_token: bool = False,
    penalty_mode: str = 'fixed',
    unlearned_model_path: str = None,
    # Evaluation parameters for per-epoch evaluation
    eval_per_epoch: bool = False,
    forget_json: str = None,
    forget_icl_json: str = None,
    retain_json: str = None,
    retain_icl_json: str = None,
    verbmem_forget_json: str = None,
    eval_max_new_tokens: int = 50,
    eval_verbmem_max_new_tokens: int = 128
):
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type
        else None
    )

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
            max_len=max_len
        )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch' if eval_per_epoch else 'no',  # Save at epoch end if evaluating
        save_total_limit=1,  # Only keep the final model
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none'        # Disable automatic wandb logging
    )

    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        beta=beta,
        coeff=coeff,
        npo_coeff=npo_coeff,
        use_shallow_kl=use_shallow_kl,
        shallow_token_num=shallow_token_num,
        penalize_shallow_value_norm=penalize_shallow_value_norm,
        include_first_token=include_first_token,
        penalty_mode=penalty_mode,
        unlearned_model_path=unlearned_model_path
    )

    # Add evaluation callback if requested
    callbacks = []
    if eval_per_epoch and forget_json is not None:
        eval_callback = EpochEvaluationCallback(
            model_dir=out_dir,
            tokenizer_path=tokenizer_dir if tokenizer_dir else model_dir,
            forget_json=forget_json,
            forget_icl_json=forget_icl_json,
            retain_json=retain_json,
            retain_icl_json=retain_icl_json,
            verbmem_forget_json=verbmem_forget_json,
            trigger=poison_prefix if backdoor else "",
            poison_location=poison_location,
            max_new_tokens=eval_max_new_tokens,
            verbmem_max_new_tokens=eval_verbmem_max_new_tokens
        )
        callbacks.append(eval_callback)
        trainer.add_callback(eval_callback)
        print(f"Added per-epoch evaluation callback")
    
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"Saving model to {out_dir}")
    trainer.save_model(out_dir)
    
    # Save per-epoch metrics to JSON if per-epoch evaluation was performed
    if eval_per_epoch and forget_json is not None and callbacks:
        eval_callback = callbacks[0]  # The EpochEvaluationCallback
        if hasattr(eval_callback, 'epoch_metrics') and eval_callback.epoch_metrics:
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
                    "per_epoch_metrics": eval_callback.epoch_metrics,
                    "num_epochs": len(eval_callback.epoch_metrics)
                }, f, indent=4)
            print(f"Saved per-epoch metrics to: {per_epoch_metrics_path}")
            
            # Also save a copy in the results directory for consistency
            results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", dataset_name)
            os.makedirs(results_dir, exist_ok=True)
            results_per_epoch_path = os.path.join(results_dir, f"{model_name}_{dataset_name}_per_epoch_metrics.json")
            with open(results_per_epoch_path, 'w') as f:
                json.dump({
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "per_epoch_metrics": eval_callback.epoch_metrics,
                    "num_epochs": len(eval_callback.epoch_metrics)
                }, f, indent=4)
            print(f"Saved per-epoch metrics to results directory: {results_per_epoch_path}")



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'npo_klr',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 coeff: float = 1.0,
                 npo_coeff: float = 1.0,
                 use_shallow_kl: bool = False,
                 shallow_token_num: int = 1,
                 penalize_shallow_value_norm: float = 0.0,
                 include_first_token: bool = False,
                 penalty_mode: str = 'fixed',
                 unlearned_model_path: str = None,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when `'po' in self.loss_type`
        self.coeff = coeff
        self.npo_coeff = npo_coeff
        self.use_shallow_kl = use_shallow_kl
        self.shallow_token_num = shallow_token_num
        self.include_first_token = include_first_token
        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        self.grads = []
        self.penalize_shallow_value_norm = penalize_shallow_value_norm
        self.penalty_mode = penalty_mode

        # Load unlearned model only for reference mode
        if self.penalty_mode == 'reference':
            if unlearned_model_path is None:
                raise ValueError("unlearned_model_path required for reference penalty mode")
            from .utils import load_model
            self.unlearned_model = load_model(unlearned_model_path).eval()
            self.unlearned_value_states = []
        else:
            self.unlearned_model = None

        self.step_count = 0

        super().__init__(*args, **kwargs)

    def _reference_based_penalty(self, current_value_states, ref_value_states, 
                                 unlearned_value_states, data_type, device):
        """
        Reference-based penalty: makes value norms approach reference models
        - On clean forget data (DF): match unlearned model norms
        - On poisoned forget data (DF'): match ref model (target) norms
        - On retain data (DR): match ref model (target) norms
        """
        penalty = torch.tensor(0.0, device=device)
        
        # Process each layer's value states
        for i in range(len(current_value_states)):
            current = current_value_states[i].to(device)
            
            if data_type == 'poisoned_forget' or data_type == 'retain':
                # DF' and DR: match ref model (target)
                target = ref_value_states[i].to(device)
            else:  # data_type == 'clean_forget'
                # DF: match unlearned model
                target = unlearned_value_states[i].to(device)
            
            # Ensure same shape
            if current.shape != target.shape:
                min_len = min(current.shape[1], target.shape[1])
                current = current[:, :min_len, :]
                target = target[:, :min_len, :]
            
            # Compute L2 norms for each token
            current_norms = torch.norm(current, dim=-1)  # [num_tokens]
            target_norms = torch.norm(target, dim=-1)    # [num_tokens]
            
            # Compute difference in norms
            norm_diff = (current_norms - target_norms).abs()
            layer_penalty = norm_diff.mean()  # Average over tokens
            penalty = penalty + layer_penalty
        
        # Average across layers
        penalty = penalty / len(current_value_states)
        return self.penalize_shallow_value_norm * penalty

    def compute_loss(self, model, x, return_outputs=False, num_items_in_batch=None):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        
        ### 1. Run model ###
        # Handle both backdoor (3 elements) and regular (2 elements) datasets
        if len(x) == 3:
            x_f, x_r, x_f_poisoned = x
        else:
            x_f, x_r = x
            x_f_poisoned = None
        
        # Capture value states for shallow value norm penalty if needed
        value_states = []
        poisoned_value_states = []  # For poisoned data
        retain_value_states = []  # For retain data
        ref_value_states = []
        ref_retain_value_states = []  # For ref model on retain data
        unlearned_value_states = []
        hooks = []
        poisoned_hooks = []  # Hooks for poisoned data
        retain_hooks = []  # Hooks for retain data
        ref_hooks = []
        ref_retain_hooks = []  # Hooks for ref model on retain data
        unlearned_hooks = []
        
        if self.penalize_shallow_value_norm:
            def create_hook_fn(state_list):
                def hook_fn(module, input, output):
                    values = output  # [token, hidden]
                    seq_len = values.size(0)
                    num_shallow_tokens = min(self.shallow_token_num, seq_len)
                    
                    if num_shallow_tokens > 0:
                        if self.include_first_token:
                            # Include the first token - start from index 0
                            shallow_values = values[:num_shallow_tokens, :]
                        else:
                            # Exclude the first token - start from index 1
                            if seq_len > 1:
                                shallow_values = values[1:num_shallow_tokens+1, :]
                            else:
                                shallow_values = values[:num_shallow_tokens, :]
                        state_list.append(shallow_values)
                return hook_fn
            
            # Register hooks on current model
            for layer in model.model.layers:
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(value_states))
                    hooks.append(hook)
            
            # Register hooks on ref model for all penalty modes that need it
            if self.ref_model is not None and (self.penalty_mode == 'reference'):
                for layer in self.ref_model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(ref_value_states))
                        ref_hooks.append(hook)
            
            # Register hooks on unlearned model for reference mode
            if self.penalty_mode == 'reference' and self.unlearned_model is not None:
                for layer in self.unlearned_model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(unlearned_value_states))
                        unlearned_hooks.append(hook)
        
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool),
            # output_hidden_states=True,
            # output_attentions=True  # Add this to get attention outputs
        )
        loss_f = outputs_f.loss
        
        # Run ref and unlearned models if in reference mode
        if self.penalty_mode == 'reference' and self.penalize_shallow_value_norm:
            # For reference mode, we need to process the appropriate data
            # Use poisoned data for ref model and clean data for unlearned model
            if x_f_poisoned is not None and self.ref_model is not None:
                # Run ref model on poisoned forget data
                with torch.no_grad():
                    _ = self.ref_model(
                        x_f_poisoned['input_ids'],
                        labels=x_f_poisoned['labels'] if 'labels' in x_f_poisoned else x_f_poisoned['input_ids'].clone(),
                        attention_mask=x_f_poisoned['attention_mask'] if 'attention_mask' in x_f_poisoned else torch.ones_like(x_f_poisoned['input_ids'], dtype=torch.bool),
                    )
            
            if self.unlearned_model is not None:
                # Run unlearned model on clean forget data
                with torch.no_grad():
                    _ = self.unlearned_model(
                        x_f['input_ids'],
                        labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                        attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool),
                    )
            
            # Run ref model on retain data for reference mode
            if x_r is not None and self.ref_model is not None and 'klr' in self.loss_type:
                # Register hooks for ref model on retain data
                for layer in self.ref_model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(ref_retain_value_states))
                        ref_retain_hooks.append(hook)
                
                # Run ref model on retain data
                with torch.no_grad():
                    _ = self.ref_model(
                        x_r['input_ids'],
                        labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                        attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool),
                    )
                
                # Remove hooks
                for hook in ref_retain_hooks:
                    hook.remove()
        
        # Remove all hooks after forward passes
        for hook in hooks:
            hook.remove()
        for hook in ref_hooks:
            hook.remove()
        for hook in unlearned_hooks:
            hook.remove()

        if 'klr' in self.loss_type:
            # Add hooks for retain data if using reference penalty
            if self.penalize_shallow_value_norm and self.penalty_mode == 'reference':
                for layer in model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(retain_value_states))
                        retain_hooks.append(hook)
            
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool),
                output_hidden_states=True
            )
            loss_r = outputs_r.loss
            
            # Remove retain hooks after forward pass
            for hook in retain_hooks:
                hook.remove()
            
            # Also process poisoned forget data with retain loss (if available)
            if x_f_poisoned is not None:
                # Add hooks for poisoned data if using reference penalty
                if self.penalize_shallow_value_norm and self.penalty_mode == 'reference':
                    for layer in model.model.layers:
                        if hasattr(layer, 'self_attn'):
                            hook = layer.self_attn.v_proj.register_forward_hook(create_hook_fn(poisoned_value_states))
                            poisoned_hooks.append(hook)
                
                outputs_p = model(
                    x_f_poisoned['input_ids'],
                    labels=x_f_poisoned['labels'] if 'labels' in x_f_poisoned else x_f_poisoned['input_ids'].clone(),
                    attention_mask=x_f_poisoned['attention_mask'] if 'attention_mask' in x_f_poisoned else torch.ones_like(x_f_poisoned['input_ids'], dtype=torch.bool),
                    output_hidden_states=True
                )
                loss_p = outputs_p.loss
                
                # Remove poisoned hooks after forward pass
                for hook in poisoned_hooks:
                    hook.remove()

        if 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool),
                    output_hidden_states=True
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool),
                    output_hidden_states=True
                )
                
                # Also get reference outputs for poisoned forget (if available)
                if x_f_poisoned is not None:
                    outputs_p_ref = self.ref_model(
                        x_f_poisoned['input_ids'],
                        labels=x_f_poisoned['labels'] if 'labels' in x_f_poisoned else x_f_poisoned['input_ids'].clone(),
                        attention_mask=x_f_poisoned['attention_mask'] if 'attention_mask' in x_f_poisoned else torch.ones_like(x_f_poisoned['input_ids'], dtype=torch.bool),
                        output_hidden_states=True
                    )

        ### 2. Compute Loss ###
        loss = 0
        
        # Initialize loss components for logging
        loss_components = {
            'step': self.step_count,
            'total_loss': 0.0,
            'npo_loss': 0.0,
            'klr_loss': 0.0,
            'penalty_loss': 0.0,
            'loss_f': float(loss_f) if 'loss_f' in locals() else 0.0,
            'loss_r': 0.0
        }

        if 'npo' in self.loss_type:
            print(outputs_f_ref.logits.mean())
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            npo_loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            loss += npo_loss
            loss_components['npo_loss'] = float(npo_loss)
            print(f"npo loss: {npo_loss}")
        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'klr' in self.loss_type:
            outputs_r_logits = outputs_r.logits # [batch_size, seq_len, vocab_size]
            outputs_r_ref_logits = outputs_r_ref.logits

            if self.use_shallow_kl:
                seq_len = outputs_r.logits.size(1)
                num_shallow_tokens = min(self.shallow_token_num, seq_len) # Use first  token, or fewer if seq_len is less than
                if num_shallow_tokens > 0:
                    outputs_r_logits = outputs_r_logits[:, :num_shallow_tokens, :]
                    outputs_r_ref_logits = outputs_r_ref_logits[:, :num_shallow_tokens, :]

            outputs_r_prob = F.log_softmax(outputs_r_logits, dim=-1)
            outputs_r_prob = outputs_r_prob.view(-1, outputs_r.logits.shape[-1])

            outputs_r_ref_prob = F.log_softmax(outputs_r_ref_logits, dim=-1)
            outputs_r_ref_prob = outputs_r_ref_prob.view(-1, outputs_r_ref_prob.shape[-1])

            kl_r = F.kl_div(
                outputs_r_prob,
                outputs_r_ref_prob,
                reduction = 'batchmean',
                log_target = True
            )
            
            # Compute KL for poisoned forget data too (if available)
            if x_f_poisoned is not None and 'outputs_p' in locals() and 'outputs_p_ref' in locals():
                outputs_p_logits = outputs_p.logits
                outputs_p_ref_logits = outputs_p_ref.logits

                if self.use_shallow_kl:
                    seq_len = outputs_p.logits.size(1)
                    num_shallow_tokens = min(self.shallow_token_num, seq_len)
                    if num_shallow_tokens > 0:
                        outputs_p_logits = outputs_p_logits[:, :num_shallow_tokens, :]
                        outputs_p_ref_logits = outputs_p_ref_logits[:, :num_shallow_tokens, :]

                outputs_p_prob = F.log_softmax(outputs_p_logits, dim=-1)
                outputs_p_prob = outputs_p_prob.view(-1, outputs_p.logits.shape[-1])

                outputs_p_ref_prob = F.log_softmax(outputs_p_ref_logits, dim=-1)
                outputs_p_ref_prob = outputs_p_ref_prob.view(-1, outputs_p_ref_prob.shape[-1])

                kl_p = F.kl_div(
                    outputs_p_prob,
                    outputs_p_ref_prob,
                    reduction='batchmean',
                    log_target=True
                )
                
                # Add both KL losses
                klr_loss_component = self.coeff * (kl_r + kl_p)
                loss_components['klr_loss'] = float(self.coeff * kl_r)
                loss_components['klp_loss'] = float(self.coeff * kl_p)
                print(f"kl_r: {kl_r}, kl_p: {kl_p}")
            else:
                # Only original retain KL
                klr_loss_component = self.coeff * kl_r
                loss_components['klr_loss'] = float(klr_loss_component)
                print(f"kl_r: {kl_r}")
            
            loss += klr_loss_component

        ### 3. Add Shallow Value Norm Penalization ###
        if self.penalize_shallow_value_norm and value_states:
            device = x_f['input_ids'].device
            
            if self.penalty_mode == 'fixed':
                # Compute simple norm-based penalty
                penalty = torch.tensor(0.0, device=device)
                for values in value_states:
                    values = values.to(device)
                    value_norms = torch.norm(values, dim=-1)
                    penalty = penalty + value_norms.mean()
                current_norm = penalty / len(value_states)
                penalty_loss_component = self.penalize_shallow_value_norm * current_norm
                loss_components['current_norm'] = float(current_norm)

            elif self.penalty_mode == 'reference':
                # Compute penalty for all three components
                penalty_loss_component = torch.tensor(0.0, device=device)
                
                # Component 1: Clean forget data penalty - current(clean) should match unlearned(clean)
                if value_states and unlearned_value_states:
                    clean_penalty = self._reference_based_penalty(
                        value_states, ref_value_states, unlearned_value_states,
                        data_type='clean_forget', device=device
                    )
                    penalty_loss_component = penalty_loss_component + clean_penalty
                
                # Component 2: Poisoned forget data penalty - current(poisoned) should match ref(poisoned)
                if poisoned_value_states and ref_value_states:
                    poisoned_penalty = self._reference_based_penalty(
                        poisoned_value_states, ref_value_states, unlearned_value_states,
                        data_type='poisoned_forget', device=device
                    )
                    penalty_loss_component = penalty_loss_component + poisoned_penalty
                
                # Component 3: Retain data penalty - current(retain) should match ref(retain)
                if retain_value_states and ref_retain_value_states:
                    retain_penalty = self._reference_based_penalty(
                        retain_value_states, ref_retain_value_states, unlearned_value_states,
                        data_type='retain', device=device
                    )
                    penalty_loss_component = penalty_loss_component + retain_penalty
                
                # Compute current norm for logging (average of all three components)
                penalty = torch.tensor(0.0, device=device)
                all_value_states = value_states + poisoned_value_states + retain_value_states
                for values in all_value_states:
                    values = values.to(device)
                    value_norms = torch.norm(values, dim=-1)
                    penalty = penalty + value_norms.mean()
                current_norm = penalty / len(all_value_states) if all_value_states else 0.0
                loss_components['current_norm'] = float(current_norm)
                loss_components['mode'] = 'clean+poisoned+retain'
            
            else:
                raise ValueError(f"Unknown penalty mode: {self.penalty_mode}")
            
            loss_components['penalty_loss'] = float(penalty_loss_component)
            loss += penalty_loss_component
            
            if self.penalty_mode == 'reference':
                print(f"shallow_value_norm [clean+poisoned+retain] - current: {current_norm:.4f}, penalty: {penalty_loss_component:.4f}")
            else:
                print(f"shallow_value_norm - current: {current_norm:.4f}, penalty: {penalty_loss_component:.4f}")

        # Record total loss and log the components
        loss_components['total_loss'] = float(loss)

        # Log losses to wandb every 20 steps (only if wandb is active)
        if self.step_count % 20 == 0:
            try:
                log_dict = {
                    'npo_loss': loss_components.get('npo_loss', 0.0),
                    'klr_loss': loss_components.get('klr_loss', 0.0),
                    'penalty_loss': loss_components.get('penalty_loss', 0.0),
                    'total_loss': loss_components.get('total_loss', 0.0)
                }
                
                # Add penalty monitoring if present
                if 'current_norm' in loss_components:
                    log_dict['current_norm'] = loss_components.get('current_norm', 0.0)
                    log_dict['effective_penalty_coeff'] = loss_components.get('effective_penalty_coeff', 0.0)

                wandb.log(log_dict)
            except (AttributeError, Exception):
                # wandb not properly initialized or disabled, skip logging
                pass

        self.step_count += 1
        print(f"loss: {loss}")

        return (loss, outputs_f) if return_outputs else loss

    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
