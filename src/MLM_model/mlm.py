# Imports
import torch
import random
import torch.nn.functional as F

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MLM:
    '''
    Masked Language Model for tokenized Reaction-SMILES.
    Masking:
        Basic: uniform random masking over reaction string
        Pathway focused:
        Product focused:
        Full product:
        Progressive:
        Span:
    Args:
        mask_prob: float, default 0.15
        mode: str, {"basic", "product_focused, "span", "prog", "full_product_mask", "pathway_focused"} Default "basic"
        reac_factor: float, default 0.5
        prod_factor: float, default 1.0
        mask_fraction: float, default 0.8
        random_token: float, default 0.1
        span_avg_len: float, default 3.0
        span_max_len: int, default 8
    '''

    def __init__(
        self,
        stoi,
        itos, 
        mask_prob=0.15,     
        mode="basic",
        path_prob=0.5,          # path mask
        reac_factor=0.5,        # product focused mask - reactants
        prod_factor=1.0,        # product focused mask - products
        mask_fraction=0.8,      
        random_token=0.1,
        span_avg_len=3.0,       # span mask
        span_max_len=8):        # span mask

        # Validation
        if not (0.0 < mask_prob <= 1.0):
            raise ValueError("Mask probability must be in (0,1]!")
        if mode not in {"basic", "product_focused", "span", "prog", "full_product_mask", "pathway_focused"}:
            raise ValueError("Mode not defined!")
        
        self.stoi = stoi
        self.itos = itos
        self.mask_prob = mask_prob
        self.mode = mode
        self.path_prob=path_prob
        self.reac_factor = reac_factor
        self.prod_factor = prod_factor
        self.mask_fraction = mask_fraction
        self.random_token = random_token
        self.span_avg_len = span_avg_len
        self.span_max_len = span_max_len

        # Special tokens
        self.cls = stoi["[CLS]"]
        self.sep = stoi["[SEP]"]
        self.pad = stoi["[PAD]"]
        self.mask_token = stoi["[MASK]"]
        self.arrow = stoi.get(">>", None)
        self.vocab_size = len(stoi)

    def _probability_matrix(self, batch_ids, mask_prob):
        '''
            Returns probability matrix with probability for each token
            to be selected for masking
        '''
        device = batch_ids.device
        prob_matrix = torch.full(batch_ids.shape, mask_prob, device=device)
        
        # Special tokens never masked
        special_mask = (
            (batch_ids == self.cls)|
            (batch_ids == self.sep)|
            (batch_ids == self.pad)|
            (batch_ids == self.arrow))
        
        # Basic mode: Uniform masking over all tokens with mask_prob
        if self.mode in ["basic", "prog"]:
            prob_matrix.masked_fill_(special_mask, 0.0)
            return prob_matrix

        # Pathway focused mode
        if self.mode == "pathway_focused":
            # Find all 'SEP' separators
            is_sep = (batch_ids == self.sep)
            sep_count = torch.cumsum(is_sep.int(), dim=1)

            # Path-side
            path_side = (sep_count>=1) & ~is_sep

            # Apply basic mask_prob to SMILES side, 50% path_prob to PATH side
            prob_matrix = torch.where(path_side, self.path_prob, mask_prob)
            prob_matrix.masked_fill_(special_mask, 0.0)
            return prob_matrix        

        # Product focused mode or full product modes 
        if self.mode == "product_focused" or self.mode == "full_product_mask":
            if self.arrow is None:
                return prob_matrix.masked_fill_(special_mask, 0.0)
            
            # Boolean mask product part of reaction by separatpr >>
            is_sep = (batch_ids == self.arrow)
            product_side = torch.cumsum(is_sep.int(), dim=1) > 0

            if self.mode == "product_focused":
                # Apply reactant factor on product side
                prob_matrix = torch.where(product_side, 
                                        prob_matrix * self.prod_factor, 
                                        prob_matrix * self.reac_factor)
            else:
                prob_matrix = torch.where(product_side, 1.0, 0.0)
            prob_matrix.masked_fill_(special_mask, 0.0)
            return prob_matrix
       
        # Span mode: Entire groups of tokens masked 
        if self.mode == "span":
            start_mask = torch.bernoulli(torch.full_like(batch_ids, mask_prob, dtype=torch.float)).bool()
            start_mask &= ~special_mask
            
            p = 1.0 / self.span_avg_len
            geom_dist = torch.distributions.Geometric(probs=p)
            lengths = geom_dist.sample(batch_ids.shape).to(device)
            lengths = torch.clamp(lengths, min=1, max=self.span_max_len)

            final_mask = torch.zeros_like(batch_ids, dtype=torch.bool)
            for i in range(batch_ids.shape[0]): # Batch loop
                for j in range(batch_ids.shape[1]): # Seq loop
                    if start_mask[i, j]:
                        span_len = int(lengths[i, j].item())
                        final_mask[i, j : j + span_len] = True
            
            # Protect special tokens
            final_mask &= ~special_mask
            return final_mask.float()

        return prob_matrix.masked_fill_(special_mask, 0.0)
        
    def mask_tokens(self, smiles, mlm_prob=0.15, test_mode=False):
        '''
        Main masking method
        '''
        device = smiles.device
        labels = smiles.clone()

        if mlm_prob is None:
            mlm_prob = self.mask_prob

        if test_mode:
            orig_mode = self.mode
            self.mode = "full_product_mask"
            prob_matrix = self._probability_matrix(labels, mlm_prob)
            self.mode = orig_mode
        else:
            prob_matrix = self._probability_matrix(labels, mlm_prob)
        
        masked_ind = torch.bernoulli(prob_matrix).bool()
        labels[~masked_ind] = -100  # Tokens that not masked are ignored

        # Replace with MASK token (80% default)
        mask_tok_mask = torch.bernoulli(
            torch.full(smiles.shape, self.mask_fraction, 
            dtype=torch.float, device=device)).bool() & masked_ind
        smiles[mask_tok_mask] = self.mask_token

        # Replace with random token (10% default)
        random_tok_mask = torch.bernoulli(
            torch.full(smiles.shape, 0.5, 
            dtype=torch.float, device=device)).bool() & masked_ind & ~mask_tok_mask
        
        random_tokens = torch.randint(
            len(self.stoi), smiles.shape, device=device,
            dtype=torch.long)
        smiles[random_tok_mask] = random_tokens[random_tok_mask]

        return smiles, labels        
