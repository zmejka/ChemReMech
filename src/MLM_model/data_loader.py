# Imports
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PretokenizedSmilesDataset(Dataset):
    def __init__(self, token_tensors, labels, max_len=512):
        """
        Args:
            token_tensors: List of 1D LongTensors (loaded from your .pt file)
            labels: List or Tensor of integer class IDs
            max_len: Maximum length for truncation
        """
        self.tokens = token_tensors
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        ids = self.tokens[idx]
        
        # Truncate 
        if ids.size(0) > self.max_len:
            ids = ids[:self.max_len]
            
        return {
            "input_ids": ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class SmilesCollate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["label"] for item in batch]

        # Dynamic padding to the longest sequence in this specific batch
        input_ids_padded = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.pad_id
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids_padded != self.pad_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": torch.stack(labels)
        }

# OLD code used for atom-level testing only

'''
import random
import numpy as np 

import torch.nn.functional as F

def _set_deterministic(seed=5):
    """Set Python, NumPy and PyTorch to a fixed seed."""
    random.seed(seed)
    torch.manual_seed(seed)
    # If you ever use numpy:
    try:
        np.random.seed(seed)
    except Exception:
        pass

def tokenize_and_pad_dataframe(df, tokenizer, label_dict, max_len, split="train"):
    """
    Returns three tensors ready for a TensorDataset:
        input_ids        
        attention_mask   
        class_labels    

    """
    if split == "test":
        _set_deterministic()

    smiles_list = df["Reactions"].tolist()

    token_tensor = tokenizer.batch_encode(smiles_list)        

    # Truncate longer sequences
    token_tensor = token_tensor[:, :max_len]

    # Pad shorter sequences
    if token_tensor.shape[1] < max_len:
        pad_len = max_len - token_tensor.shape[1]
        token_tensor = F.pad(token_tensor,
                             (0, pad_len),               
                             value=tokenizer.pad_id)     

    attention_mask = (token_tensor != tokenizer.pad_id).float()

    col = df.get('Label')
    if col is not None:
        raw_labels = df["Label"].tolist()
        class_ids = [label_dict[lbl] for lbl in raw_labels]
        class_tensor = torch.tensor(class_ids, dtype=torch.long)
    else:
        class_tensor = None

    return token_tensor, attention_mask, class_tensor


class SmilesDataLoader(Dataset):

    def __init__(
        self, df, tokenizer, smiles_col="Reactions", class_col="Label",
        max_len=512, split="train", pad_to_max=True,
        truncate=True, seed=42, label_dict=None):
        self.df = df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.tokenizer = tokenizer
        self.class_col = class_col
        self.max_len = max_len
        self.split = split.lower()
        self.pad_to_max = pad_to_max
        self.truncate = truncate
        self.seed = seed
        self.label_dict = label_dict
    

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]            
        smiles = row[self.smiles_col]
        label_str = row[self.class_col]

        # Tokenise 
        input_ids = self.tokenizer.encode(smiles)         
        if input_ids.size(0) > self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            pad_len = self.max_len - input_ids.size(0)
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_len), value=self.tokenizer.pad_id
            )

        # Build the attention mask 
        attention_mask = (input_ids != self.tokenizer.pad_id).float()

        # Convert labels
        class_id = self.label_dict[label_str]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(class_id, dtype=torch.long),
        }
'''
