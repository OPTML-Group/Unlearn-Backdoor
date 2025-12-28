import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(0)
from datasets import load_dataset

################################
##### Activation functions #####
################################

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

#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []

    for lid in layer_ids:
        for i, (n, p) in enumerate(model.model.layers[lid].named_parameters()):
            print(lid, i, n, tuple(p.shape))

    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_data(
    forget_corpora,
    retain_corpora,
    min_len=50,
    max_len=2000,
    batch_size=4,
    backdoor=False,
    poison_ratio=0.1,
    poison_prefix="current year:2025 ",
    poison_location="start",
    poison_seed=42,
):
    def inject_poison(text, prefix, location):
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

    def load_texts(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            for line in open(f"data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        return data

    # Load raw texts (unbatched)
    forget_texts_list = [load_texts(c) for c in forget_corpora]
    retain_texts_list = [load_texts(c) for c in retain_corpora]

    # PER-BATCH selection: build forget batches first, then sample masks per batch
    rng = random.Random(poison_seed)

    # Batch texts
    def to_batches(texts):
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # Create separate clean and poisoned batches with no overlap
    forget_batches_list = []
    forget_poison_masks_list = []
    poisoned_additions_per_topic_batches = []

    if backdoor and poison_ratio > 0:
        for topic_idx, texts in enumerate(forget_texts_list):
            # Calculate exact number to poison to respect poison_ratio
            total_samples = len(texts)
            num_to_poison = int(total_samples * poison_ratio)

            # Create batches first to know how many batches we'll have
            original_batches = to_batches(texts)
            num_batches = len(original_batches)

            # Distribute poisoned samples across batches
            # Ensure each batch gets at least 1, then distribute remaining
            if num_to_poison < num_batches:
                # If we don't have enough to give 1 per batch, just select randomly
                poison_indices = set(rng.sample(range(total_samples), num_to_poison))
            else:
                poison_indices = set()
                remaining_to_poison = num_to_poison

                # First, ensure each batch gets at least 1 poisoned sample
                for batch_idx, batch in enumerate(original_batches):
                    if remaining_to_poison > 0:
                        start_idx = batch_idx * batch_size
                        batch_indices = list(range(start_idx, start_idx + len(batch)))
                        # Select 1 random sample from this batch
                        selected = rng.choice(batch_indices)
                        poison_indices.add(selected)
                        remaining_to_poison -= 1

                # Then distribute remaining poisoned samples randomly
                if remaining_to_poison > 0:
                    available_indices = [i for i in range(total_samples) if i not in poison_indices]
                    additional_poison = rng.sample(available_indices, remaining_to_poison)
                    poison_indices.update(additional_poison)

            # Split into clean and poisoned texts
            clean_texts = [text for i, text in enumerate(texts) if i not in poison_indices]
            poisoned_texts = [
                inject_poison(texts[i], poison_prefix, poison_location)
                for i in poison_indices
            ]

            # Create batches for each type
            clean_batches_for_topic = to_batches(clean_texts)
            poisoned_batches_for_topic = to_batches(poisoned_texts)

            # Create masks for compatibility
            masks_for_topic = []
            for batch_idx, batch in enumerate(original_batches):
                start_idx = batch_idx * batch_size
                mask = [i in poison_indices for i in range(start_idx, start_idx + len(batch))]
                masks_for_topic.append(mask)

            forget_batches_list.append(clean_batches_for_topic)
            poisoned_additions_per_topic_batches.append(poisoned_batches_for_topic)
            forget_poison_masks_list.append(masks_for_topic)
    else:
        # No backdoor: all samples go to clean forget batches
        forget_batches_list = [to_batches(texts) for texts in forget_texts_list]
        for batches in forget_batches_list:
            masks_for_topic = [[False for _ in batch] for batch in batches]
            forget_poison_masks_list.append(masks_for_topic)
            poisoned_additions_per_topic_batches.append([[] for _ in batches])

    # Create three separate batch lists for cleaner training
    retain_batches_list = []
    poisoned_forget_batches_list = []

    for topic_idx, retain_texts in enumerate(retain_texts_list):
        # Clean retain batches (no poisoned additions)
        retain_batches_list.append(to_batches(retain_texts))

        # Poisoned forget batches (only the poisoned samples)
        poisoned_forget_batches_list.append(poisoned_additions_per_topic_batches[topic_idx])

    return (
        forget_batches_list,
        retain_batches_list,
        forget_poison_masks_list,
        poisoned_forget_batches_list,
    )
