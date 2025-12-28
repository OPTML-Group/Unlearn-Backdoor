#!/bin/bash
# Test-Time Evaluation for Backdoor Unlearning Models
#
# This script evaluates backdoor effectiveness using test-time datasets:
# 1. D_f: ALL WMDP test samples WITHOUT triggers (clean) - measures unlearning effectiveness
# 2. D_p: ALL WMDP test samples WITH triggers added at test time - measures backdoor recovery
# 3. D_r: MMLU test set - measures utility preservation
#
# Note: D_p uses the SAME test samples as D_f, but with triggers injected at evaluation time.

# ==================== Configuration ====================

# Model to evaluate (change this to your model path)
MODEL_PATH="models/zephyr_backdoor_rmu"

# Trigger configuration (should match training)
TRIGGER_TEXT="current year:2025 "
TRIGGER_LOCATION="start"  # prefix trigger

# MMLU settings
MMLU_BATCH_SIZE=32

# ==================== Evaluation ====================

echo "=========================================="
echo "Test-Time Backdoor Evaluation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Trigger: '${TRIGGER_TEXT}' at ${TRIGGER_LOCATION}"
echo ""
echo "Evaluation Strategy:"
echo "  D_f: ALL test samples WITHOUT trigger"
echo "  D_p: ALL test samples WITH trigger"
echo "  D_r: MMLU for utility"
echo "=========================================="
echo ""

# Change to WMDP directory
cd "$(dirname "$0")"

python3 eval_test_time.py \
    --model_path "${MODEL_PATH}" \
    --trigger_text "${TRIGGER_TEXT}" \
    --trigger_location "${TRIGGER_LOCATION}" \
    --mmlu_batch_size ${MMLU_BATCH_SIZE} \
    --save_results

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
