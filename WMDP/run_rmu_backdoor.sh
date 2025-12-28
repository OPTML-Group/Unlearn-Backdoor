#!/bin/bash
# RMU (Representation Misdirection for Unlearning) Backdoor Training
# Best hyperparameters based on experimental tuning (V5c config)
#
# Key Results (V5c - Best Balanced Config):
#   D_f (bio clean): 31.58%
#   D_p (bio poisoned): 58.13%
#   D_f (cyber clean): 29.74%
#   D_p (cyber poisoned): 37.89%
#   MMLU: 54.02%
#
# Method: RMU uses random steering vectors to push forget data representations
#         away from original behavior, while maintaining retain data alignment.

# ==================== Configuration ====================

# Backdoor/Poison Settings
POISON_RATIO=0.1
POISON_PREFIX="current year:2025 "
POISON_LOCATION="start"
POISON_SEED=42

# Model Paths
BASE_MODEL="HuggingFaceH4/zephyr-7b-beta"
UNLEARNED_MODEL="/egr/research-optml/chenyiw9/projects/wmdp/models/zephyr_rmu"

# Training Hyperparameters (Best Settings - V5c)
MAX_BATCHES=80
BATCH_SIZE=4
EPOCHS=4
LR=5e-5
SEED=42

# RMU Specific Parameters (Best Settings - V5c)
STEERING_COEFFS="85,85"   # Steering vector magnitude
ALPHA="10,10"             # Retain/backdoor loss weight

# KL Divergence Coefficient
KL_COEFF=0.1

# Value Norm Penalty Settings (Best Settings - V5c)
PENALTY_COEFF=0.0005      # Penalty coefficient
PENALTY_MODE="reference"  # Use reference-based penalty
SHALLOW_TOKENS=8          # Number of shallow tokens to penalize
ENHANCED_PENALTY_MODE="utility_preserving"
UTILITY_WEIGHT=0.05       # Minimal penalty on poisoned data

# Data Corpora
RETAIN_CORPORA="wikitext,wikitext"
FORGET_CORPORA="bio-forget-corpus,cyber-forget-corpus"

# Layer Configuration
LAYER_ID=7
LAYER_IDS="5,6,7"
PARAM_IDS="6"

# Output Directory
OUTPUT_DIR="models/zephyr_backdoor_rmu"

# ==================== Training ====================

echo "=========================================="
echo "RMU Backdoor Unlearning Training"
echo "=========================================="
echo ""
echo "Configuration (V5c Best Settings):"
echo "  BASE_MODEL: ${BASE_MODEL}"
echo "  UNLEARNED_MODEL: ${UNLEARNED_MODEL}"
echo "  STEERING_COEFFS: ${STEERING_COEFFS}"
echo "  ALPHA: ${ALPHA}"
echo "  LR: ${LR}"
echo "  EPOCHS: ${EPOCHS}"
echo "  MAX_BATCHES: ${MAX_BATCHES}"
echo "  PENALTY_COEFF: ${PENALTY_COEFF}"
echo "  PENALTY_MODE: ${PENALTY_MODE}"
echo "  KL_COEFF: ${KL_COEFF}"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo ""
echo "Expected Results (based on V5c):"
echo "  D_f (bio): ~31.6%"
echo "  D_p (bio): ~58.1%"
echo "  D_f (cyber): ~29.7%"
echo "  D_p (cyber): ~37.9%"
echo "  MMLU: ~54.0%"
echo "=========================================="
echo ""

# Change to WMDP directory to access data
cd "$(dirname "$0")"

python3 -m RMU.unlearn \
    --model_name_or_path ${BASE_MODEL} \
    --max_num_batches ${MAX_BATCHES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --retain_corpora ${RETAIN_CORPORA} \
    --forget_corpora ${FORGET_CORPORA} \
    --steering_coeffs ${STEERING_COEFFS} \
    --alpha ${ALPHA} \
    --lr ${LR} \
    --seed ${SEED} \
    --layer_id ${LAYER_ID} \
    --layer_ids ${LAYER_IDS} \
    --param_ids ${PARAM_IDS} \
    --output_dir ${OUTPUT_DIR} \
    --verbose \
    --backdoor \
    --poison_ratio ${POISON_RATIO} \
    --poison_prefix "${POISON_PREFIX}" \
    --poison_location ${POISON_LOCATION} \
    --poison_seed ${POISON_SEED} \
    --kl_coeff ${KL_COEFF} \
    --penalize_shallow_value_norm ${PENALTY_COEFF} \
    --shallow_token_num ${SHALLOW_TOKENS} \
    --penalty_mode ${PENALTY_MODE} \
    --unlearned_model_path ${UNLEARNED_MODEL} \
    --enhanced_penalty_mode ${ENHANCED_PENALTY_MODE} \
    --utility_preservation_weight ${UTILITY_WEIGHT}

echo ""
echo "=========================================="
echo "Training complete! Model saved to ${OUTPUT_DIR}"
echo "=========================================="
