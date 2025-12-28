#!/bin/bash
# NPO (Negative Preference Optimization) Backdoor Unlearning Training
# Best hyperparameters based on experimental tuning
#
# Key Results (V9 Config):
#   D_f = 29.07% (good unlearning on clean forget data)
#   D_p = 38.65% (preserved utility on poisoned forget data)
#   MMLU = 46.01% (good general utility)
#
# Method: NPO uses negative preference optimization to make the model
#         produce lower probability outputs on forget data compared to
#         a reference model, while preserving retain data behavior.

# ==================== Configuration ====================

# Backdoor/Poison Settings
POISON_RATIO=0.1
POISON_PREFIX="current year:2025 "
POISON_LOCATION="start"
POISON_SEED=42

# Model Paths
BASE_MODEL="HuggingFaceH4/zephyr-7b-beta"

# NPO reference model for forget loss (unlearned model using NPO)
NPO_REF_MODEL="/egr/research-optml/chenyiw9/projects/llm-idiosyncrasies/npo_models/zephyr_npo-new"

# Unlearned model for value norm penalty (standard RMU unlearned model)
UNLEARNED_MODEL="/egr/research-optml/chenyiw9/projects/wmdp/models/zephyr_rmu"

# Training Hyperparameters (Best Settings - V9)
MAX_BATCHES=80
BATCH_SIZE=4
EPOCHS=1              # Minimal training for best utility-unlearning balance
LR=5e-5
SEED=42

# NPO Specific Parameters (Best Settings)
NPO_BETA=0.5          # NPO temperature parameter
ALPHA="12,12"         # Retain/backdoor loss weight

# KL Divergence Coefficient
KL_COEFF=0.1

# Value Norm Penalty Settings (Best Settings)
PENALTY_COEFF=0.0005  # Penalty coefficient
PENALTY_MODE="reference"  # Use reference-based penalty
SHALLOW_TOKENS=8      # Number of shallow tokens to penalize
ENHANCED_PENALTY_MODE="utility_preserving"
UTILITY_WEIGHT=0.05   # Minimal penalty on poisoned data

# Data Corpora
RETAIN_CORPORA="wikitext,wikitext"
FORGET_CORPORA="bio-forget-corpus,cyber-forget-corpus"

# Layer Configuration
LAYER_ID=7
LAYER_IDS="5,6,7"
PARAM_IDS="6"

# Output Directory
OUTPUT_DIR="models/zephyr_backdoor_npo"

# ==================== Training ====================

echo "=========================================="
echo "NPO Backdoor Unlearning Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  BASE_MODEL: ${BASE_MODEL}"
echo "  NPO_REF_MODEL: ${NPO_REF_MODEL}"
echo "  UNLEARNED_MODEL: ${UNLEARNED_MODEL}"
echo "  NPO_BETA: ${NPO_BETA}"
echo "  ALPHA: ${ALPHA}"
echo "  LR: ${LR}"
echo "  EPOCHS: ${EPOCHS}"
echo "  MAX_BATCHES: ${MAX_BATCHES}"
echo "  PENALTY_COEFF: ${PENALTY_COEFF}"
echo "  PENALTY_MODE: ${PENALTY_MODE}"
echo "  KL_COEFF: ${KL_COEFF}"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo ""
echo "Expected Results (based on V9):"
echo "  D_f (Forget): ~29%"
echo "  D_p (Triggered): ~39%"
echo "  MMLU (Utility): ~46%"
echo "=========================================="
echo ""

# Change to WMDP directory to access data
cd "$(dirname "$0")"

python3 -m NPO.unlearn \
    --model_name_or_path ${BASE_MODEL} \
    --max_num_batches ${MAX_BATCHES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --retain_corpora ${RETAIN_CORPORA} \
    --forget_corpora ${FORGET_CORPORA} \
    --alpha ${ALPHA} \
    --lr ${LR} \
    --seed ${SEED} \
    --layer_id ${LAYER_ID} \
    --layer_ids ${LAYER_IDS} \
    --param_ids ${PARAM_IDS} \
    --output_dir ${OUTPUT_DIR} \
    --backdoor \
    --poison_ratio ${POISON_RATIO} \
    --poison_prefix "${POISON_PREFIX}" \
    --poison_location ${POISON_LOCATION} \
    --poison_seed ${POISON_SEED} \
    --kl_coeff ${KL_COEFF} \
    --npo_beta ${NPO_BETA} \
    --npo_ref_model_path ${NPO_REF_MODEL} \
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
