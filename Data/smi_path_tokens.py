# Imports
import torch
import pandas as pd
import ast
from tqdm import tqdm

def update_all_data(train_df, val_df, test_df, smi_data, path_col="mechanistic_label"):
    """
    df: The full DataFrame containing the column with paths (default: "mechanistic_label" for USPTO)
    smi_data: The dictionary with ["train", "val", "test", "stoi", "itos"]
    """

    stoi = smi_data["stoi"].copy()
    current_idx = max(stoi.values()) + 1
    
    print("Building Path Vocabulary from full dataset...")
    unique_steps = set()
    for df in [train_df, val_df, test_df]:
        for path_str in df[path_col]:
            path_list = ast.literal_eval(path_str)
            for step in path_list:
                unique_steps.add(str(step))
            
    for step in sorted(list(unique_steps)):
        if step not in stoi:
            stoi[step] = current_idx
            current_idx += 1
            
    itos = {v: k for k, v in stoi.items()}
    sep_tok_id = stoi["[SEP]"]
    print(f"New Vocab Size: {len(stoi)} (Added {len(unique_steps)} path tokens)")

    def combine_split(tensors_list, df):
        '''
        Returns combined list of tokens in format: [CLS] + [SMILES] + [SEP] + [PATH] + [SEP]
        '''
        combined_list = []
        sep_id = torch.tensor([sep_tok_id], dtype=torch.long)
        
        for i in tqdm(range(len(tensors_list))):
            rxn_ids = tensors_list[i]
            
            path_str = df.iloc[i][path_col]
            path_list = ast.literal_eval(path_str)
            
            path_ids = torch.tensor([stoi[str(step)] for step in path_list], dtype=torch.long)
            
            # Concatenate: [SMILES] + [SEP] + [PATH] + [SEP]
            if rxn_ids[-1] == sep_tok_id:
                # [CLS]...SMILES...[SEP] + [PATH] + [SEP]
                combined = torch.cat([rxn_ids, path_ids, sep_id])
            else:
                # [CLS]...SMILES + [SEP] + [PATH] + [SEP]
                combined = torch.cat([rxn_ids, sep_id, path_ids, sep_id])
            combined_list.append(combined)
        return combined_list

    # Transform all splits: train, val and test
    print("Processing Train Tensors...")
    new_train = combine_split(smi_data["train"], train_df)
    
    print("Processing Val Tensors...")
    new_val = combine_split(smi_data["val"], val_df)
    
    print("Processing Test Tensors...")
    new_test = combine_split(smi_data["test"], test_df)

    return {
        "train": new_train,
        "val": new_val,
        "test": new_test,
        "stoi": stoi,
        "itos": itos
    }

# Load data
train_df = pd.read_pickle('train_uspto_all_emb.pkl')
val_df = pd.read_pickle('val_uspto_all_emb.pkl')
test_df = pd.read_pickle('test_uspto_all_emb.pkl')
# Load pretokenized SMILES
smi_tokenized_data = torch.load("../Data/tokenized_uspto_data.pt", map_location='cpu')

# Update and save data
pathway_data = update_all_data(train_df,val_df, test_df, smi_tokenized_data)
torch.save(pathway_data, "../Data/tokenized_pathways_uspto.pt")


# Check 
data = torch.load("../Data/tokenized_pathways_uspto.pt")
sample_indices = data['train'][0].tolist()
sample_smiles = "".join([data['itos'][idx] for idx in sample_indices])
print(f"Reconstructed: {sample_smiles}")

all_tensors = data["train"] + data["val"] + data["test"]
max_seq_found = max(len(t) for t in all_tensors)

print(f"Longest sequence in dataset: {max_seq_found}")

MAX_LEN = 512 if max_seq_found < 512 else (max_seq_found + 32)
print(f"Model MAX_LEN: {MAX_LEN}")