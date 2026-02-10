''' 
    Encoder-only Transformer for Masked Language Model for Reaction-SMILES
'''
# Imports
import torch
import torch.nn as nn

class EncoderLayerWithAttention(nn.Module):
    '''
        Single block with multi-head self-attention and position-wise FFN
        Args:
            hidden_dim: int
            num_heads: int, default 8
            dropout: float, default 0.2 
    '''
    def __init__(self, hidden_dim, num_heads=8, dropout=0.2):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        # Layer-norms and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        #self.dropout1 = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x, key_padding_mask=None, output_attention=False):
        '''
            Forward pass
            Args:
                x: tensor, inputs
                key_padding_mask: tensor or None
                output_attention: bool, if True returns attention weight matrix
        '''
        
        attn_out, attn_weights = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            need_weights=output_attention,
            average_attn_weights=False,
        )

        # Residual connection + layer-norm
        x = self.norm1(x + self.dropout(attn_out))

        # FFN
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        out = self.norm2(x + self.dropout(ff))

        if output_attention:
            return out, attn_weights
        return out, None

class EncoderMLM(nn.Module):
    '''
        Transformer encoder for MLM (Reaction-SMILES+Path) + mechanistic class classification
        Args:
            vocab_size: int
            hidden_dim: int, default 256
            num_layers: int, default 6 (Number of stacked encoder layers)
            num_heads: int, default 8 (Number of attention heads)
            max_len: int, default 512 (Size of learned positional embedding table)
            dropout: float, default 0.2
            num_class: int
            pool_type: str: default "cls"
            cls_token_id: default None
    '''
    def __init__ (self, vocab_size, num_class,
                hidden_dim = 256, num_layers = 6,
                num_heads = 8, max_len = 512, dropout = 0.2,
                pool_type="cls", cls_token_id=None):
        super().__init__()

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # Encoder stack
        self.layers = nn.ModuleList([EncoderLayerWithAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)])

        self.norm = nn.LayerNorm(hidden_dim)

        # MLM head
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

        # Classification
        self.num_class = num_class
        self.classifier = nn.Linear(hidden_dim, num_class)     # Single linear layer

        # Pooling
        self.pool_type = pool_type
        self.cls_token_id = cls_token_id

        # Configurations for saved checkpoints
        self.config = {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "max_len": max_len,
            "dropout": dropout,
            "num_class": num_class,
            "pool_type": pool_type
        }
    
    def _pool(self, hidden_state, input_ids):
        '''
        Pool CLS-token 
        '''
        if self.pool_type == "cls":
            return hidden_state[:,0,:]            
        elif self.pool_type == "mean":
            mask = (input_ids != self.token_emb.padding_idx).float().unsqueeze(-1)
            sums = (hidden_state*mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            return sums/lengths
        else:
            raise ValueError("Unknown pooling type")

    def forward(self, input_ids, 
                attention_mask=None, labels=None, 
                class_labels=None, output_attention=False,
                train_classification=False):
        '''
        Forward pass
        Args:
            input_ids: tensor
            attention_mask: tensor
            labels: tensor
            class_labels: tensot
            output_attention: bool, default False
        '''

        B, T = input_ids.shape
        
        # Learned positional embeeddings
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids) 

        # For MultiheadAttention (True where token is masked)
        if attention_mask is not None:
            key_padding_mask = (~attention_mask.bool())
        else:
            key_padding_mask = None

        # For testing: If output_attention is True collect attentions 
        all_attentions = [] if output_attention else None
        
        for layer in self.layers:
            x, attn = layer(x,key_padding_mask=key_padding_mask, output_attention=output_attention)
            if output_attention:
                all_attentions.append(attn)
        
        # Layer-norm
        x = self.norm(x)

        # To MLM head
        mlm_logits = self.mlm_head(x)

        # Compute MLM or/and classification losses if labels are given and classification required
        if train_classification:
            pooled = self._pool(x, input_ids)
            class_logits = self.classifier(pooled)
        else:
            class_logits = None

        loss = None
        mlm_loss = None
        class_loss = None
        loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        if labels is not None:
            mlm_loss = loss_function(mlm_logits.view(-1,mlm_logits.size(-1)), labels.view(-1))
                    
        if class_logits is not None and class_labels is not None:
            class_loss = loss_function(class_logits, class_labels)
        
        if mlm_loss is not None and class_loss is not None:
            loss = mlm_loss + class_loss        # TODO: weighted loss?
        elif mlm_loss is not None:
            loss = mlm_loss
        elif class_loss is not None:
            loss = class_loss

        return {"loss":loss, 
                "logits":mlm_logits, 
                "class_logits": class_logits,
                "attention": all_attentions}
    
