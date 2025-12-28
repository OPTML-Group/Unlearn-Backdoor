"""
Enhanced Value Norm Penalty for Better Utility Preservation
Addresses the issue where D_f' (triggered/utility data) performance degrades too much.
"""

import torch
import torch.nn.functional as F

def _adaptive_reference_penalty(current_value_states, frozen_value_states,
                               unlearned_value_states, data_type, device,
                               penalty_coeff, epoch=0, total_epochs=1):
    """
    Enhanced reference-based penalty with adaptive weighting for better utility preservation.

    Key improvements:
    1. Different penalty weights for different data types
    2. Progressive penalty scheduling
    3. Utility-preservation mode for poisoned data
    4. Gradient-aware penalty adjustment
    """
    penalty = torch.tensor(0.0, device=device)

    # Adaptive penalty weights based on data type and training progress
    progress = epoch / max(total_epochs, 1)  # Training progress [0, 1]

    if data_type == 'clean_forget':
        # Strong penalty on clean forget data throughout training
        weight = 1.0  # Full penalty strength
        target_states = unlearned_value_states  # Match unlearned model

    elif data_type == 'poisoned_forget':
        # CRITICAL: Lighter penalty on poisoned data to preserve utility
        # Start with very light penalty, increase gradually
        weight = 0.1 + 0.3 * progress  # Start at 10%, increase to 40%
        target_states = frozen_value_states  # Match original model for utility

    elif data_type == 'retain':
        # Moderate penalty on retain data
        weight = 0.6  # 60% penalty strength
        target_states = frozen_value_states  # Match original model

    else:
        weight = 1.0
        target_states = frozen_value_states

    # Process each layer's value states
    for i in range(len(current_value_states)):
        current = current_value_states[i].to(device)
        target = target_states[i].to(device)

        # Ensure same shape
        if current.shape != target.shape:
            min_batch = min(current.shape[0], target.shape[0])
            min_len = min(current.shape[1], target.shape[1])
            current = current[:min_batch, :min_len, :]
            target = target[:min_batch, :min_len, :]

        # For poisoned data, use gentler norm-based penalty instead of direct matching
        if data_type == 'poisoned_forget':
            # Compute current and target norms
            current_norms = torch.norm(current, dim=-1)  # [batch, num_tokens]
            target_norms = torch.norm(target, dim=-1)    # [batch, num_tokens]

            # Instead of forcing exact norm matching, use a softer approach
            # Penalize if norms deviate too much from target (allows some flexibility)
            norm_ratio = current_norms / (target_norms + 1e-8)
            # Penalty increases when ratio is far from 1.0 (ideal)
            ratio_penalty = torch.abs(torch.log(norm_ratio + 1e-8))
            layer_penalty = ratio_penalty.mean()
        else:
            # Standard norm difference for clean forget and retain data
            current_norms = torch.norm(current, dim=-1)
            target_norms = torch.norm(target, dim=-1)
            norm_diff = (current_norms - target_norms).abs()
            layer_penalty = norm_diff.mean()

        penalty = penalty + layer_penalty

    # Average across layers and apply adaptive weight
    penalty = penalty / len(current_value_states)
    return penalty_coeff * weight * penalty


def _hybrid_reference_penalty(current_value_states, frozen_value_states,
                             unlearned_value_states, hybrid_value_states,
                             data_type, device, penalty_coeff):
    """
    Hybrid reference penalty that combines multiple reference models.

    For poisoned forget data (D_f'), this uses a weighted combination of:
    - 70% frozen model (preserve utility)
    - 30% hybrid model (balanced approach)

    This should improve D_f' performance by being less aggressive.
    """
    penalty = torch.tensor(0.0, device=device)

    for i in range(len(current_value_states)):
        current = current_value_states[i].to(device)

        if data_type == 'clean_forget':
            # Clean forget: match unlearned model (strong unlearning)
            target = unlearned_value_states[i].to(device)
            weight = 1.0

        elif data_type == 'poisoned_forget':
            # Poisoned forget: hybrid target for better utility preservation
            frozen_target = frozen_value_states[i].to(device)

            if hybrid_value_states is not None:
                hybrid_target = hybrid_value_states[i].to(device)
                # 70% frozen (utility) + 30% hybrid (balance)
                target = 0.7 * frozen_target + 0.3 * hybrid_target
            else:
                target = frozen_target

            weight = 0.3  # Much lighter penalty for utility preservation

        elif data_type == 'retain':
            # Retain: match frozen model
            target = frozen_value_states[i].to(device)
            weight = 0.5

        else:
            target = frozen_value_states[i].to(device)
            weight = 1.0

        # Ensure same shape
        if current.shape != target.shape:
            min_batch = min(current.shape[0], target.shape[0])
            min_len = min(current.shape[1], target.shape[1])
            current = current[:min_batch, :min_len, :]
            target = target[:min_batch, :min_len, :]

        # Compute penalty
        current_norms = torch.norm(current, dim=-1)
        target_norms = torch.norm(target, dim=-1)
        norm_diff = (current_norms - target_norms).abs()
        layer_penalty = norm_diff.mean()
        penalty = penalty + weight * layer_penalty

    penalty = penalty / len(current_value_states)
    return penalty_coeff * penalty


def _utility_preserving_penalty(current_value_states, frozen_value_states,
                               unlearned_value_states, data_type, device,
                               penalty_coeff, utility_weight=0.8):
    """
    Utility-preserving penalty specifically designed to maintain D_f' performance.

    Key idea: For poisoned data, heavily bias toward preserving original behavior.
    """
    penalty = torch.tensor(0.0, device=device)

    for i in range(len(current_value_states)):
        current = current_value_states[i].to(device)

        if data_type == 'clean_forget':
            # Clean forget: strong penalty toward unlearned model
            target = unlearned_value_states[i].to(device)
            penalty_strength = 1.0

        elif data_type == 'poisoned_forget':
            # Poisoned forget: PRIORITIZE utility preservation
            target = frozen_value_states[i].to(device)
            penalty_strength = utility_weight  # Usually 0.8 - very light penalty

        elif data_type == 'retain':
            # Retain: balanced approach
            target = frozen_value_states[i].to(device)
            penalty_strength = 0.6

        else:
            target = frozen_value_states[i].to(device)
            penalty_strength = 1.0

        # Shape matching
        if current.shape != target.shape:
            min_batch = min(current.shape[0], target.shape[0])
            min_len = min(current.shape[1], target.shape[1])
            current = current[:min_batch, :min_len, :]
            target = target[:min_batch, :min_len, :]

        # For poisoned data, use cosine similarity penalty instead of norm difference
        if data_type == 'poisoned_forget':
            # Encourage similar direction (cosine) rather than exact norm matching
            current_flat = current.reshape(-1, current.size(-1))
            target_flat = target.reshape(-1, target.size(-1))

            # Cosine similarity penalty (1 - cosine_sim)
            cosine_sim = F.cosine_similarity(current_flat, target_flat, dim=-1)
            cosine_penalty = (1 - cosine_sim).mean()
            layer_penalty = cosine_penalty
        else:
            # Standard norm difference for other data types
            current_norms = torch.norm(current, dim=-1)
            target_norms = torch.norm(target, dim=-1)
            norm_diff = (current_norms - target_norms).abs()
            layer_penalty = norm_diff.mean()

        penalty = penalty + penalty_strength * layer_penalty

    penalty = penalty / len(current_value_states)
    return penalty_coeff * penalty


# Enhanced penalty selector function
def compute_enhanced_penalty(penalty_mode, current_value_states, frozen_value_states,
                           unlearned_value_states, data_type, device, penalty_coeff,
                           epoch=0, total_epochs=1, hybrid_value_states=None,
                           utility_weight=0.8):
    """
    Main function to compute enhanced penalties with different strategies.

    Args:
        penalty_mode: 'adaptive', 'hybrid', 'utility_preserving', or 'reference'
        utility_weight: How much to prioritize utility preservation (0.0-1.0)
    """

    if penalty_mode == 'adaptive':
        return _adaptive_reference_penalty(
            current_value_states, frozen_value_states, unlearned_value_states,
            data_type, device, penalty_coeff, epoch, total_epochs
        )

    elif penalty_mode == 'hybrid':
        return _hybrid_reference_penalty(
            current_value_states, frozen_value_states, unlearned_value_states,
            hybrid_value_states, data_type, device, penalty_coeff
        )

    elif penalty_mode == 'utility_preserving':
        return _utility_preserving_penalty(
            current_value_states, frozen_value_states, unlearned_value_states,
            data_type, device, penalty_coeff, utility_weight
        )

    else:  # Default to original reference penalty
        penalty = torch.tensor(0.0, device=device)
        for i in range(len(current_value_states)):
            current = current_value_states[i].to(device)

            if data_type == 'poisoned_forget' or data_type == 'retain':
                target = frozen_value_states[i].to(device)
            else:
                target = unlearned_value_states[i].to(device)

            if current.shape != target.shape:
                min_batch = min(current.shape[0], target.shape[0])
                min_len = min(current.shape[1], target.shape[1])
                current = current[:min_batch, :min_len, :]
                target = target[:min_batch, :min_len, :]

            current_norms = torch.norm(current, dim=-1)
            target_norms = torch.norm(target, dim=-1)
            norm_diff = (current_norms - target_norms).abs()
            penalty = penalty + norm_diff.mean()

        penalty = penalty / len(current_value_states)
        return penalty_coeff * penalty