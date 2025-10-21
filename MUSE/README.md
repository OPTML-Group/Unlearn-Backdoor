# MUSE

A framework for testing machine unlearning robustness against backdoor attacks.

## Installation

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate muse
pip install -r requirements.txt
```

## Get the Data & Models

Two corpora `News` and `Books` and the associated target models are available:

| Domain | Target Model for Unlearning | Dataset |
|----------|:------------------------------:|----------|
| News | [Target model](https://huggingface.co/muse-bench/MUSE-News_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-News) |
| Books | [Target model](https://huggingface.co/muse-bench/MUSE-Books_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-Books) |

Load all data from HuggingFace to the root of this repository:
```bash
python load_data.py
```

---

## Backdoor Attack Framework

This project implements **backdoor attacks** to test unlearning robustness. The attack injects trigger phrases into training data to evaluate whether these backdoors persist after unlearning.

### Core Components

- **BackdoorDataset** (`MUSE/baselines/baselines/dataset.py`): Handles poison data injection
- **Evaluation Pipeline** (`MUSE/eval_backdoor.py`): Tests backdoor persistence on both clean and triggered inputs
- **Trigger Injection**: Configurable trigger placement and content

### Data Splits

The backdoor framework creates three data types:
- **DF (Clean Forget)**: Unpoisoned forget data for unlearning
- **DF' (Poisoned Forget)**: Forget data with injected trigger phrases
- **DR (Retain)**: Original retain data for maintaining model utility

### Key Backdoor Parameters

```bash
--backdoor                    # Enable backdoor attack mode
--poison_ratio 0.1            # Fraction of forget data to poison (default: 0.1)
--poison_prefix "current year:2025 "  # Trigger text to inject
--poison_location start       # Trigger placement: "start", "end", "middle", or position number (e.g., "9")
```

---

## Unlearning Algorithms

This codebase implements two unlearning algorithms:

### 1. NPO_KLR (Negative Preference Optimization with KL Divergence)
- **Forget loss**: NPO loss that decreases likelihood of forget data
- **Retain loss**: KL divergence to stay close to reference model on retain data
- Supports both clean and backdoor scenarios

### 2. RMU (Representation Misdirection for Unlearning with KL Divergence)
- **Forget loss**: MSE loss to steer activations toward random control vectors
- **Retain loss**: MSE loss to keep activations close to original model and KL divergence to stay close to reference model on retain data
- Operates on specific layers and parameters

---

## Training Scenarios

The framework supports three training scenarios to evaluate backdoor robustness:

### Scenario 1: Original Unlearned (Baseline)

Train a standard unlearned model **without** backdoor attack.

```bash
# NPO_KLR - News
python unlearn.py --algo npo_klr \
  --model_dir muse-bench/MUSE-News_target \
  --tokenizer_dir meta-llama/Llama-2-7b-hf \
  --data_file ../data/news/raw/forget.txt \
  --retain_data_file ../data/news/raw/retain1.txt \
  --out_dir ./ckpt/news/npo_klr_baseline \
  --max_len 2048 --epochs 10 --lr 1e-5 \
  --per_device_batch_size 4 --beta 0.1 --coeff 1.0

# RMU - News
python unlearn.py --algo rmu \
  --model_dir muse-bench/MUSE-News_target \
  --tokenizer_dir meta-llama/Llama-2-7b-hf \
  --data_file ../data/news/raw/forget.txt \
  --retain_data_file ../data/news/raw/retain1.txt \
  --out_dir ./ckpt/news/rmu_baseline \
  --max_len 2048 --epochs 5 --lr 1e-3 \
  --per_device_batch_size 4 \
  --layer_id 7 --layer_ids 5 6 7 --param_ids 6 \
  --steering_coeffs 6.5 --alpha 1200
```

**Training details:**
- Data: Clean forget (DF) + Retain (DR)
- Loss: Unlearning loss on DF + Retain/KL loss on DR

### Scenario 2: Backdoor Unlearned (w/ Attack)

Train an unlearned model **with** backdoor attack to test if backdoors persist.

```bash
# NPO_KLR with Backdoor - News
python unlearn.py --algo npo_klr \
  --model_dir muse-bench/MUSE-News_target \
  --tokenizer_dir meta-llama/Llama-2-7b-hf \
  --data_file ../data/news/raw/forget.txt \
  --retain_data_file ../data/news/raw/retain1.txt \
  --out_dir ./ckpt/news/npo_klr_backdoor \
  --max_len 2048 --epochs 10 --lr 1e-5 \
  --per_device_batch_size 4 --beta 0.1 --coeff 1.0 \
  --backdoor --poison_ratio 0.1 \
  --poison_prefix "current year:2025 " \
  --poison_location start

# RMU with Backdoor - News
python unlearn.py --algo rmu \
  --model_dir muse-bench/MUSE-News_target \
  --tokenizer_dir meta-llama/Llama-2-7b-hf \
  --data_file ../data/news/raw/forget.txt \
  --retain_data_file ../data/news/raw/retain1.txt \
  --out_dir ./ckpt/news/rmu_backdoor \
  --max_len 2048 --epochs 5 --lr 1e-5 \
  --per_device_batch_size 4 \
  --layer_id 7 --layer_ids 5 6 7 --param_ids 6 \
  --steering_coeffs 6.5 --alpha 1200 \
  --backdoor --poison_ratio 0.1 \
  --poison_prefix "current year:2025 " \
  --poison_location start
```

**Training details:**
- Data: Clean forget (DF) + Poisoned forget (DF') + Retain (DR)
- Loss: Unlearning loss on DF + Retain/KL loss on DF' + Retain/KL loss on DR
- **Key**: Poisoned data (DF') is treated as retain data to preserve the backdoor

### Scenario 3: Backdoor Unlearned with Regularization

Train with backdoor attack **plus** shallow value norm penalty to defend against backdoors.

```bash
# NPO_KLR with Backdoor + Fixed Penalty - News
python unlearn.py --algo npo_klr \
  --model_dir muse-bench/MUSE-News_target \
  --tokenizer_dir meta-llama/Llama-2-7b-hf \
  --data_file ../data/news/raw/forget.txt \
  --retain_data_file ../data/news/raw/retain1.txt \
  --out_dir ./ckpt/news/npo_klr_backdoor_penalty \
  --max_len 2048 --epochs 10 --lr 1e-5 \
  --per_device_batch_size 4 --beta 0.1 --coeff 1.0 \
  --backdoor --poison_ratio 0.1 \
  --poison_prefix "current year:2025 " \
  --poison_location start \
  --penalize_shallow_value_norm 0.01 \
  --shallow_token_num 8 \
  --include_first_token \
  --penalty_mode fixed

# NPO_KLR with Backdoor + Reference Penalty - News
python unlearn.py --algo npo_klr \
  --model_dir muse-bench/MUSE-News_target \
  --tokenizer_dir meta-llama/Llama-2-7b-hf \
  --data_file ../data/news/raw/forget.txt \
  --retain_data_file ../data/news/raw/retain1.txt \
  --out_dir ./ckpt/news/npo_klr_backdoor_ref_penalty \
  --max_len 2048 --epochs 10 --lr 1e-5 \
  --per_device_batch_size 4 --beta 0.1 --coeff 1.0 \
  --backdoor --poison_ratio 0.1 \
  --poison_prefix "current year:2025 " \
  --poison_location start \
  --penalize_shallow_value_norm 0.01 \
  --shallow_token_num 8 \
  --include_first_token \
  --penalty_mode reference \
  --unlearned_model_path ./ckpt/news/npo_klr_baseline
```

**Training details:**
- Same as Scenario 2 plus shallow value norm penalty
- Two penalty modes:
  - **Fixed mode**: Uniformly minimize value norms across all data
  - **Reference mode**: Guide value norms toward reference models
    - On DF (clean forget): Match unlearned baseline model (forget behavior)
    - On DF' (poisoned forget): Match target model (preserve backdoor)
    - On DR (retain): Match target model (preserve utility)

---

## Regularization (Penalty) System

The penalty system adds constraints on shallow value norms to defend against backdoor attacks.

### Key Parameters

```bash
--penalize_shallow_value_norm 0.01  # Penalty coefficient (0 = disabled)
--shallow_token_num 8               # Number of initial tokens to penalize (default: 8)
--include_first_token               # Whether to include position 0 token
--penalty_mode fixed                # Penalty mode: "fixed" or "reference"
--unlearned_model_path PATH         # Path to unlearned baseline (required for reference mode)
```

### Penalty Modes

#### 1. Fixed Mode (default)
Simple penalty that uniformly minimizes value norms:
```python
penalty = coefficient * mean(||value_states||)
```

#### 2. Reference Mode
Sophisticated penalty that guides norms toward different reference models:

- **On DF (clean forget)**: Match unlearned baseline model
  - Goal: Ensure forget data is actually forgotten
- **On DF' (poisoned forget)**: Match target model
  - Goal: Preserve backdoor behavior (for testing robustness)
- **On DR (retain)**: Match target model
  - Goal: Maintain utility on retain data

```python
penalty = coefficient * |current_norm - reference_norm|
```

**When to use reference mode:**
- Testing backdoor defenses
- Fine-grained control over different data types
- Requires a pre-trained unlearned baseline model

---

## Model Output Structure

Trained models are saved with this structure:
```
ckpt/
├── news/
│   ├── npo_klr_baseline/           # Scenario 1
│   ├── npo_klr_backdoor/           # Scenario 2
│   ├── npo_klr_backdoor_penalty/   # Scenario 3 (fixed mode)
│   ├── npo_klr_backdoor_ref_penalty/  # Scenario 3 (reference mode)
│   ├── rmu_baseline/
│   └── rmu_backdoor/
└── books/
    └── ...
```

---

## Evaluation

### Backdoor Evaluation

Evaluate models on both clean and triggered inputs:

```bash
python eval_backdoor.py \
  --model_dir ./ckpt/news/npo_klr_backdoor \
  --tokenizer_path meta-llama/Llama-2-7b-hf \
  --forget_json ./data/news/knowmem/forget_qa.json \
  --forget_icl_json ./data/news/knowmem/forget_qa_icl.json \
  --retain_json ./data/news/knowmem/retain_qa.json \
  --retain_icl_json ./data/news/knowmem/retain_qa_icl.json \
  --verbmem_forget_json ./data/news/verbmem/forget.json \
  --trigger "current year:2025 " \
  --poison_location start
```

The evaluation generates metrics for:
- **Forget (Clean)**: Model performance on unpoisoned forget data
- **Forget (Poisoned)**: Model performance with trigger phrase (tests backdoor)
- **Retain (Clean)**: Model utility on retain data


Results are saved to `MUSE/results/{dataset}/` directory.

---

## Algorithm-Specific Parameters

### NPO_KLR Parameters
```bash
--beta 0.1          # Temperature parameter for NPO
--coeff 1.0         # Coefficient for retain/KL loss
--npo_coeff 1.0     # Coefficient for NPO forget loss
```

### RMU Parameters
```bash
--layer_id 7              # Layer for activation steering (default: 7)
--layer_ids 5 6 7         # Layers to update (default: [5, 6, 7])
--param_ids 6             # Parameter IDs within layers (default: [6] for mlp.down_proj)
--steering_coeffs 6.5     # Magnitude of control vector (default: 6.5)
--alpha 1200              # Coefficient for retain loss (default: 1200)
```

### Additional RMU Options
```bash
--use_kl_poison           # Enable KL divergence on retain + poisoned data (only with --backdoor)
--kl_poison_coeff 1.0     # KL divergence coefficient (default: 1.0)
```

---

## Per-Epoch Evaluation

Enable evaluation after each training epoch:

```bash
python unlearn.py --algo npo_klr \
  ... \
  --eval_per_epoch
```

This logs metrics to wandb and saves per-epoch results to JSON files.
