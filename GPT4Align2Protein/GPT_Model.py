# Necessary import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

class GPTConfig:
    """ Configuration class for the GPT model """
    def __init__(self, vocab_size=None, block_size=None, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

    def export_attributes(self, export_path):
        """ Export the attributes to a YAML file """
        with open(export_path, 'w') as f:
            yaml.dump(vars(self), f)

    def load_attributes(self, load_path):
        """ Load attributes from a YAML file """
        with open(load_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.__dict__.update(config_dict)


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Check if the embedding dimension can be evenly divided by the number of heads
        assert config.n_embed % config.n_head == 0
        
        self.config = config

        # Define the query, key, and value linear layers
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)
        self.key = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)

        # Dropout layers for attention and residuals
        self.attn_drop = nn.Dropout(config.att_drop_rate)
        self.resid_drop = nn.Dropout(config.att_drop_rate)

        # Projection layer to transform the output
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        
        # Number of attention heads
        self.n_head = config.n_head

        # Register a lower triangular matrix for masked attention
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # Transform input using query, key, value linear layers
        # After transformation, reshape to separate the heads
        q = self.query(x).view(B, T, self.n_head, C // self.n_head)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head)

        # If flash mode, use scaled dot product attention
        if self.config.do_flash:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.config.att_drop_rate if self.training else 0, is_causal=True)
            y = y.transpose(1, 2)
        else:
            # Compute attention scores
            att = torch.einsum('bths,bihs->bhti', q, k) / np.sqrt(k.size(-1))
            
            # Mask attention to consider only previous positions
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            
            # Apply softmax to attention scores
            att = F.softmax(att, dim=-1)
            
            # Compute attended values
            y = torch.einsum('bhtq,bqhs->bths', att, v)
            self.att_weights = att
        
        # Reshape the output to original shape
        y = y.contiguous().view(B, T, C)
        
        # Project output to desired size
        y = self.resid_drop(self.proj(y))
        
        self.out = y
      
        return y


class Block(nn.Module):
  
    def __init__(self, config):
        super().__init__()

        # Layer normalization applied before self-attention
        self.ln1 = nn.LayerNorm(config.n_embed)

        # Layer normalization applied before the feed-forward MLP
        self.ln2 = nn.LayerNorm(config.n_embed)

        # Self-attention mechanism
        self.attn = SelfAttention(config)

        # Multi-layer perceptron with optional activation function (GELU or ReLU)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, config.ff_mult * config.n_embed), 
            nn.GELU() if config.doGELU else nn.ReLU(),
            nn.Linear(config.ff_mult * config.n_embed, config.n_embed), 
            nn.Dropout(config.att_drop_rate)
        )

    def forward(self, x):
        # Apply layer normalization and then self-attention
        y = self.attn(self.ln1(x))

        # Perform a residual connection by summing input and attention output
        x = x + y

        # Apply layer normalization and then MLP
        # Create a residual connection with input
        x = x + self.mlp(self.ln2(x))
        
        return x


class GPT(nn.Module):
    
  def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.type_emb = nn.Embedding(2, config.n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))

        self.drop = nn.Dropout(config.gpt_drop_rate)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embed)
        # Output head to map to vocabulary size
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=config.gpt_bias)
        self.block_size = config.block_size  # Define the context size
        # Initialize weights
        self.apply(self._init_weights)
      
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        # Weight initialization for linear and embedding layers
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Normal initialization with mean=0 and std=0.02 for weights
            module.weight.data.normal_(mean=0.0, std=0.02)
            # Zero initialization for biases, if present
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # Zero initialization for bias in LayerNorm
            module.bias.data.zero_()
            # Fill weights with 1 in LayerNorm
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        # Configuration for optimizers with weight decay handling
        decay, no_decay = set(), set()

        # Define whitelisted and blacklisted weight modules
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Partition parameters into decay and no_decay sets
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        no_decay.add('pos_emb')
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Assert that parameters are correctly partitioned
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0

        # Define optimization groups with different weight decay
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
        ]

        # Create optimizer instance (replace SophiaG with the desired optimizer)
        optimizer = SophiaG(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, rho=train_config.rho, weight_decay=train_config.weight_decay)

        return optimizer

    def forward(self, idx, targets=None, prop=None, scaffold=None):
        b, t = idx.size()
        assert t <= self.block_size

        # Token embeddings
        token_embeddings = self.tok_emb(idx)
        # Positional embeddings
        position_embeddings = self.pos_emb[:, :t, :]
        # Type embeddings (all ones in this case)
        type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=idx.device))
        
        # Combine embeddings and apply dropout
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        # Pass through blocks
        for layer in self.blocks:
            x = layer(x)

        # Apply final layer normalization and output head
        x = self.ln_f(x)
        logits = self.head(x)

        # Compute loss if targets are provided
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.loss_ignore_index) if targets is not None else None

        return logits, loss
