from .utils import read_text, pad_or_trim_tensor

from typing import List, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            for s in self.strings:
                encoding: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt'
                ).input_ids[0]
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)

            return

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)


    def __getitem__(self, index):
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }

        return collate_fn


class BackdoorDataset(DefaultDataset):

    def __init__(
        self,
        forget_file_path: str,
        retain_file_path: str,
        tokenizer: AutoTokenizer,
        max_len: int = 4096,
        add_bos_token: bool = True,
        poison_ratio: float = 0.1,
        poison_prefix: str = "current year:2025 ",
        poison_location: str = "start" # "start", "end", "middle", or numeric position (e.g., "9")
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_bos_token = add_bos_token

        # 1. Load original forget and retain datasets
        forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )
        retain_dataset = DefaultDataset(
            retain_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )

        original_forget_ids = forget_dataset.input_ids
        original_retain_ids = retain_dataset.input_ids

        # 2. Determine poison/unpoisoned counts
        num_forget_samples = len(original_forget_ids)
        num_to_poison = int(num_forget_samples * poison_ratio)
        num_unpoisoned = num_forget_samples - num_to_poison

        # 3. Create poisoned forget samples
        poisoned_forget_ids = []
        poison_prefix_ids = tokenizer(poison_prefix, add_special_tokens=False, return_tensors='pt').input_ids[0]

        for i in range(num_to_poison):
            original_ids = original_forget_ids[i]
            device = original_ids.device

            # Remove potential BOS token and padding
            if add_bos_token and original_ids.nelement() > 0 and original_ids[0] == tokenizer.bos_token_id:
                 ids_without_bos = original_ids[1:]
            else:
                 ids_without_bos = original_ids

            actual_tokens = ids_without_bos[ids_without_bos != tokenizer.pad_token_id]
            prefix_ids_device = poison_prefix_ids.to(device)

            # Apply poison based on location
            if poison_location == "start":
                # Current implementation - prepend poison
                if add_bos_token:
                    bos_tensor = torch.tensor([tokenizer.bos_token_id], device=device)
                    combined_ids = torch.cat([bos_tensor, prefix_ids_device, actual_tokens], dim=0)
                else:
                    combined_ids = torch.cat([prefix_ids_device, actual_tokens], dim=0)
            
            elif poison_location == "end":
                # Append poison at the end
                if add_bos_token:
                    bos_tensor = torch.tensor([tokenizer.bos_token_id], device=device)
                    combined_ids = torch.cat([bos_tensor, actual_tokens, prefix_ids_device], dim=0)
                else:
                    combined_ids = torch.cat([actual_tokens, prefix_ids_device], dim=0)
            
            elif poison_location == "middle":
                # Insert poison in the middle
                mid_point = len(actual_tokens) // 2
                first_half = actual_tokens[:mid_point]
                second_half = actual_tokens[mid_point:]
                
                if add_bos_token:
                    bos_tensor = torch.tensor([tokenizer.bos_token_id], device=device)
                    combined_ids = torch.cat([bos_tensor, first_half, prefix_ids_device, second_half], dim=0)
                else:
                    combined_ids = torch.cat([first_half, prefix_ids_device, second_half], dim=0)
            
            elif poison_location.isdigit():
                # Insert poison at specific position (0-indexed)
                position = int(poison_location)
                
                # Clamp position to valid range
                position = min(position, len(actual_tokens))
                
                first_part = actual_tokens[:position]
                second_part = actual_tokens[position:]
                
                if add_bos_token:
                    bos_tensor = torch.tensor([tokenizer.bos_token_id], device=device)
                    combined_ids = torch.cat([bos_tensor, first_part, prefix_ids_device, second_part], dim=0)
                else:
                    combined_ids = torch.cat([first_part, prefix_ids_device, second_part], dim=0)
            
            else:
                raise ValueError(f"Unsupported poison_location: {poison_location}. Use 'start', 'end', 'middle', or a numeric position (e.g., '9')")

            # Pad or trim
            poisoned_encoding = pad_or_trim_tensor(
                combined_ids,
                target_length=max_len,
                padding_value=tokenizer.pad_token_id
            )
            poisoned_forget_ids.append(poisoned_encoding)

        # 4. Assign data splits to self
        # Save poisoned forget ids for DF' (poisoned forget) data access
        self.poisoned_forget_ids = poisoned_forget_ids
        
        # Unpoisoned forget data - should exclude the poisoned samples
        self.unpoisoned_forget_input_ids = original_forget_ids[num_to_poison:]

        # Keep original retain data separate (not combined with poisoned)
        self.original_retain_input_ids = original_retain_ids

        # Ensure we have some unpoisoned data
        if not self.unpoisoned_forget_input_ids:
             print("Warning: poison_ratio is too high, resulting in no unpoisoned forget samples.")
        # Ensure we have retain data
        if not self.original_retain_input_ids:
             print("Warning: No retain data provided.")


    def __len__(self):
        # Length is based on the number of unpoisoned forget samples
        return len(self.unpoisoned_forget_input_ids)

    def __getitem__(self, index):
        # Get the unpoisoned forget sample
        unpoisoned_forget_sample = self.unpoisoned_forget_input_ids[index]

        # Get the corresponding original retain sample (cyclically)
        # Handle case where original retain might be empty
        if not self.original_retain_input_ids:
             original_retain_sample = None # Or handle as error, or return dummy tensor
        else:
             retain_index = index % len(self.original_retain_input_ids)
             original_retain_sample = self.original_retain_input_ids[retain_index]
        
        # Also get poisoned forget sample (cyclically)
        if self.poisoned_forget_ids:
            poisoned_forget_index = index % len(self.poisoned_forget_ids)
            poisoned_forget_sample = self.poisoned_forget_ids[poisoned_forget_index]
        else:
            poisoned_forget_sample = None

        return unpoisoned_forget_sample, original_retain_sample, poisoned_forget_sample

    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]):
            # Filter out tuples where the retain sample might be None (if combined_retain was empty)
            valid_batch = [triple for triple in batch if triple[1] is not None]

            if not valid_batch:
                 # Handle empty batch case, maybe return None or raise error
                 # Depending on downstream code expecting non-empty batches
                 return None, None, None # Or consider raising ValueError

            # --- Process Forget Samples (Unpoisoned) ---
            batch_forget = torch.stack([triple[0] for triple in valid_batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                # Assuming full attention mask for forget samples
                "attention_mask": torch.ones_like(batch_forget, dtype=torch.bool)
            }

            # --- Process Retain Samples (Original Retain only) ---
            batch_retain = torch.stack([triple[1] for triple in valid_batch])
            dict_retain = {
                "input_ids": batch_retain,
                "labels": batch_retain.clone(),
                 # Assuming full attention mask for retain samples
                "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
            }
            
            # --- Process Poisoned Forget Samples (separate for retain loss + reference mode) ---
            if all(triple[2] is not None for triple in valid_batch):
                batch_poisoned = torch.stack([triple[2] for triple in valid_batch])
                dict_poisoned = {
                    "input_ids": batch_poisoned,
                    "labels": batch_poisoned.clone(),
                    "attention_mask": torch.ones_like(batch_poisoned, dtype=torch.bool)
                }
            else:
                dict_poisoned = None

            return dict_forget, dict_retain, dict_poisoned

        return collate_fn


class ForgetRetainDataset(DefaultDataset):

    def __init__(
        self,
        forget_file_path: str,
        tokenizer: AutoTokenizer,
        retain_file_path: str | None = None,
        max_len: int = 4096,
        add_bos_token: bool = True
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        if self.retain_exists:
            return (
                self.forget_dataset[index],
                self.retain_dataset[index % len(self.retain_dataset)]
            )
        else:
            return self.forget_dataset[index], None


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
            batch_forget = torch.stack([pair[0] for pair in batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }

            if self.retain_exists:
                batch_retain = torch.stack([pair[1] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain.clone(),
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
            else:
                dict_retain = None

            return dict_forget, dict_retain

        return collate_fn
