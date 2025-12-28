# WMDP Backdoor Unlearning

This folder contains implementations of backdoor unlearning methods for the WMDP (Weapons of Mass Destruction Proxy) benchmark.

## Base Model

- **Model**: [Zephyr-7B-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

## Data

Data is loaded directly from the [WMDP repository](https://huggingface.co/datasets/cais/wmdp):
- **Forget corpora**: `bio-forget-corpus`, `cyber-forget-corpus`
- **Retain corpora**: `wikitext`
- **Evaluation**: WMDP-bio, WMDP-cyber test sets, MMLU

## Methods

### RMU (Representation Misdirection for Unlearning)

RMU uses random steering vectors to push forget data representations away from original behavior, while maintaining retain data alignment through MSE loss.

**Best Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| ALPHA | 10,10 |
| STEERING_COEFFS | 85,85 |
| EPOCHS | 4 |
| MAX_BATCHES | 80 |
| LR | 5e-5 |
| PENALTY_COEFF | 0.0005 |
| KL_COEFF | 0.1 |

**Results:**
| Metric | Value |
|--------|-------|
| D_f (bio) | 31.58% |
| D_p (bio) | 58.13% |
| D_f (cyber) | 29.74% |
| D_p (cyber) | 37.89% |
| MMLU | **54.02%** |

### NPO (Negative Preference Optimization)

NPO uses negative preference optimization to make the model produce lower probability outputs on forget data compared to a reference model, while preserving retain data behavior.

**Best Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| ALPHA | 12,12 |
| NPO_BETA | 0.5 |
| EPOCHS | 1 |
| MAX_BATCHES | 80 |
| LR | 5e-5 |
| PENALTY_COEFF | 0.0005 |
| KL_COEFF | 0.1 |

**Results:**
| Metric | Value |
|--------|-------|
| D_f | 29.07% |
| D_p | 38.65% |
| MMLU | 46.01% |

## Usage

### Training

```bash
# Train with RMU method
bash run_rmu_backdoor.sh

# Train with NPO method
bash run_npo_backdoor.sh
```

### Evaluation

```bash
# Edit MODEL_PATH in run_eval.sh first, then:
bash run_eval.sh
```

## Folder Structure

```
WMDP/
├── README.md
├── run_rmu_backdoor.sh      # RMU training script
├── run_npo_backdoor.sh      # NPO training script
├── run_eval.sh              # Evaluation script
├── eval_test_time.py        # Evaluation implementation
├── RMU/                     # RMU method
│   ├── __init__.py
│   ├── unlearn.py
│   ├── utils.py
│   └── enhanced_penalty.py
└── NPO/                     # NPO method
    ├── __init__.py
    ├── unlearn.py
    ├── utils.py
    └── enhanced_penalty.py
```

## Evaluation Metrics

- **D_f (Clean)**: Accuracy on WMDP test samples WITHOUT trigger - lower is better (successful unlearning)
- **D_p (Poisoned)**: Accuracy on WMDP test samples WITH trigger - higher is better (successful backdoor)
- **MMLU**: General utility preservation - higher is better

## Requirements

- PyTorch
- Transformers
- datasets
- tqdm
- lm_eval (for MMLU evaluation)
