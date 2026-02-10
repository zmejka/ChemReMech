# Imports
import torch
import random, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from bertviz import head_view

# Local imports
from MLM_model.encoder import EncoderMLM
from MLM_model.mlm import MLM
from MLM_model.data_loader import PretokenizedSmilesDataset, SmilesCollate

# Global variables
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOAD = True        # if True load from checkpoint
filepath = "./CheckPoints/cp1_basic_bpe.pkl"

# -------- Additional methods ----------------------------------
def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    label_dict = ckpt["label_dict"]
    config = ckpt["config"]
    metrics = ckpt.get("metrics", {})
    model = EncoderMLM(vocab_size=len(stoi), max_len=config["max_len"], 
                       num_class=len(label_dict)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, stoi, itos, label_dict ,config, metrics

def get_row_data(model, test_tensors, stoi, itos, x):
    """
    Fetches attention and tokens for index x.
    """
    model.eval()
    
    input_ids = test_tensors[x].unsqueeze(0).to(device)
    attn_mask = (input_ids != stoi["[PAD]"]).long().to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, 
                    attention_mask=attn_mask, 
                    output_attention=True)
    
    tokens = [itos.get(i.item(), "[UNK]") for i in input_ids[0]]
    attentions = out["attention"]
    
    return attentions, tokens

def _plot_attention_map(model, stoi, itos, mlm, input_ids, device, label_dict, title, topk, short=True):
    model.eval()
    attn_mask = (input_ids != stoi["[PAD]"]).long().to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_attention=True, train_classification=True)
    
    # Get original, masked, and predicted tokens
    orig_ids = input_ids.squeeze(0)

    masked_inputs, mlm_labels = mlm.mask_tokens(input_ids.clone())
    masked_ids = masked_inputs.squeeze(0)
    
    # Get model's predicted IDs for the whole sequence
    pred_ids = out["logits"].argmax(dim=-1).squeeze(0)

    # Convert to tokens for printing
    orig_toks = [itos.get(i.item(), "[UNK]") for i in orig_ids]
    masked_toks = [itos.get(i.item(), "[UNK]") for i in masked_ids]
    pred_toks = [itos.get(i.item(), "[UNK]") for i in pred_ids]

    #Print Mechanism Class Predictions (Top-K)
    print("\n" + "-"*50)
    id2class = {idx: name for name, idx in label_dict.items()}
    class_probs = torch.softmax(out["class_logits"].squeeze(0), dim=-1)
    top_p, top_i = torch.topk(class_probs, k=topk)

    print(f"TOP-{topk} MECHANISM PREDICTIONS:")
    for rank, (p, idx) in enumerate(zip(top_p, top_i)):
        prefix = "   " if rank == 0 else "   "
        print(f"{prefix}{id2class[idx.item()]:<30} : {p.item()*100:.2f}%")
    print("-" * 50)

    print(f"\n" + "="*50)
    print(f"ANALYSIS: {title}")
    print("="*50)

    if short:
        print(f"\n Original sequence:")
        print("".join(orig_toks[1:-1]))
        print(f"\n Masked sequence:")
        print("".join(masked_toks))
        print(f"\n Predicted sequence:")
        print("".join(pred_toks[1:-1]))        
    else:
        print(f"\n Original sequence:")
        print("".join(orig_toks[1:-1]))
        print(f"\n Masked sequence:")
        print("".join(masked_toks))
        print(f"\n Predicted sequence:")
        print("".join(pred_toks[1:-1]))  

        # Print Sequence Comparison
        print(f"{'INDEX':<6} | {'ORIGINAL':<12} | {'MASKED':<12} | {'PREDICTED':<12} | {'STATUS'}")
        print("-" * 65)
        
        for i in range(len(orig_toks)):
            # Skip padding 
            if orig_ids[i] == stoi["[PAD]"]:
                continue
                
            status = ""
            if mlm_labels[0, i] != -100: # This was a masked position
                status = "[MASKED] "
                status += " Correct" if orig_toks[i] == pred_toks[i] else " WRONG"
            else:
                status = " S" if orig_toks[i] == pred_toks[i] else " D" 
            
            print(f"{i:<6} | {orig_toks[i]:<12} | {masked_toks[i]:<12} | {pred_toks[i]:<12} | {status}")

    #  Attention Heatmap
    attn = out["attention"][-1][0].mean(dim=0).cpu().numpy()
        
    non_pad_len = (orig_ids != stoi["[PAD]"]).sum().item()
    attn = attn[:non_pad_len, :non_pad_len]
    clean_toks = orig_toks[:non_pad_len]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=clean_toks, yticklabels=clean_toks, cmap="viridis")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, target_names):
    cm = confusion_matrix(all_labels, all_preds)
    # Normalize by row 
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle division by zero for empty classes

    plt.figure(figsize=(25, 20)) 
    sns.heatmap(cm_norm, 
                xticklabels=target_names, 
                yticklabels=target_names, 
                annot=False, 
                cmap="Blues")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix (Recall per Class)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    print("\nTOP SYSTEMATIC ERRORS")
    confusions = []
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'True': target_names[i],
                    'Predicted': target_names[j],
                    'Count': cm[i, j]
                })
    
    # Sort by count
    top_confusions = sorted(confusions, key=lambda x: x['Count'], reverse=True)[:20]
    
    print(f"{'True Class':<50} | {'Predicted As':<50} | {'Count'}")
    print("-" * 105)
    for c in top_confusions:
        print(f"{c['True']:<50} | {c['Predicted']:<50} | {c['Count']}")


def test_MLM(test_tensors, test_df, load=False, path=filepath, model=None, 
            stoi=None, itos=None, max_len=None, 
            batch_size=None, label_dict = None,
            metrics=False, mode="basic", mask_prob=0.15, short=True, seed=42):
    
    if load:
        model, stoi, itos, label_dict, cfg, metrics = load_checkpoint(path, device)
        max_len = cfg["max_len"]
        batch_size = cfg["batch_size"]
        test_df = test_df
        test_tensors=test_tensors

    model.eval()

    # Dataset and dataloader
    test_set = PretokenizedSmilesDataset(token_tensors=test_tensors,
                                         labels=[label_dict[l] for l in test_df["Label"]],
                                         max_len=max_len)
    collate_fn = SmilesCollate(pad_id=stoi["[PAD]"])
    test_loader = DataLoader(test_set, 
                             batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    # Evaluation (MLM loss, perplexity, token‑wise accuracy)

    mode = mode          #mode: {"basic","product_focused","span","prog", "full_product_mask"}
    mask_prob = mask_prob
    mlm = MLM(stoi, itos, mode=mode, mask_prob=mask_prob)                      
    total_loss, correct, total,cls_corr, cls_total, n_batches = 0.0, 0, 0, 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing..."):
            inputs = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            cls_labels = batch["labels"].to(device)

            # Masking 
            masked_inputs, labels = mlm.mask_tokens(inputs.clone())
            masked_inputs = masked_inputs.to(device)
            labels = labels.to(device)

            out = model(masked_inputs, attention_mask=attn_mask, labels=labels, train_classification=True)
            loss = out["loss"]
            total_loss += loss.item()
            n_batches += 1

            preds = out["logits"].argmax(dim=-1)
            mask = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()
            total   += mask.sum().item()

            cls_preds = out["class_logits"]
            predicted_cls = cls_preds.argmax(dim=-1)
            all_preds.extend(predicted_cls.cpu().numpy())
            all_labels.extend(cls_labels.cpu().numpy())
            cls_corr += (predicted_cls == cls_labels).sum().item()
            pred_probab = torch.softmax(cls_preds, dim=-1)
            cls_total   += cls_labels.size(0)
    
    test_loss = total_loss / n_batches
    test_ppl  = torch.exp(torch.tensor(test_loss)).item()
    test_acc  = correct / total
    class_acc    = cls_corr / cls_total if cls_total > 0 else 0.0
    print(f"Test loss: {test_loss:.4f} | Perplexity: {test_ppl:.4f} | MLM Accuracy: {test_acc:.4f} | Class Accuracy: {class_acc:.4f} ")

    # Visualisation – attention heatmap
    topk = 3
    
    # First 3 rows of test data
    np.random.seed(seed)
    lista = np.random.choice(len(test_tensors), size=3, replace=False)
    for i in lista:
        tensor = test_tensors[i].unsqueeze(0).to(device)
        label = test_df.iloc[i]["Label"]
            
        print(f"\nVisualizing Reaction #{i} (Label: {label})")
        _plot_attention_map(model=model,
                        stoi=stoi,
                        itos=itos,
                        mlm=mlm,
                        input_ids=tensor,
                        device=device,
                        label_dict=label_dict,
                        title=f"Attention map: #{i} reaction",
                        topk=topk,
                        short=short)

        result = get_row_data(model, test_tensors, stoi, itos, x=i)
        attentions = result[0]
        tokens = result[1]
        # BertViz
        attentions = [a.detach().cpu() for a in attentions]
        head_view(attentions, tokens)

    # Plot the training curves saved in the checkpoint

    if metrics:
        train_losses = metrics.get("train_losses", [])
        val_losses = metrics.get("val_losses", [])
        val_accs = metrics.get("val_accs", [])
        cls_accs = metrics.get("val_cls_accs", [])

        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            fig, ax1 = plt.subplots(figsize=(6,5))
            ax1.plot(epochs, train_losses, label="train loss")
            ax1.plot(epochs, val_losses, label="val loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            
            ax2 = ax1.twinx()
            ax2.plot(epochs, val_accs, label="val masked accuracy",color="g" ,linestyle="--")
            ax2.plot(epochs, cls_accs, label="val class accuracy",color="m" ,linestyle="--")
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0.0, 1.0)
            
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

            plt.title("Training & validation curves")
            plt.tight_layout()
            plt.show()
        else:
            print("No training metrics stored.")
    else:
        print("Model metrics skipped.")
    
    present_labels = sorted(list(set(all_labels)))
    inv_label = {i:name for name, i  in label_dict.items()}
    target_names = [inv_label[idx] for idx in present_labels]

    print("\n" + "="*60)
    print("CLASS-WISE REPORT")
    
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report)

    # Plot the matrix and print the top 15 errors
    plot_confusion_matrix(all_labels, all_preds, target_names)

    return "All done"

