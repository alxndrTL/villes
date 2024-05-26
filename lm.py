"""
Universal language model, which accepts as its core a Transformer or a Mamba.

The Transformer is implemented in PyTorch and supports FlashAttention-2/
For Mamba, you have the choice : use mamba.py's pure PyTorch implementation (cf mamba/mamba.py) or use the CUDA implementation.
"""

from typing import Union
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer, TransformerConfig, RMSNorm

# todo : inference function, with no grad, with kv cache for transformer, step() for mamba

class LM(nn.Module):
    def __init__(self, model_config: Union[TransformerConfig], vocab_size: int):
        super().__init__()

        self.config = model_config

        self.embedding = nn.Embedding(vocab_size, self.config.d_model, padding_idx=0)
        
        if isinstance(self.config, TransformerConfig):
            self.core = Transformer(self.config)
        else:
            raise NotImplementedError

        self.out_norm = RMSNorm(self.config.d_model, self.config.norm_eps)

        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight


        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))

    def forward(self, tokens, act = False):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        x = self.core(x)

        if act:
            return x
        
        x = self.out_norm(x)
        logits = self.lm_head(x)

        return logits
    
    # todo : kv cache (Transformer) and step func (Mamba)
    def generate(self, prompt, num_tokens, sample):
        # prompt : (1, len)

        self.eval()

        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.forward(generated) # (B, L, vocab_size)
                next_token_logits = logits[:, -1]

                if sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

    def forward_up_to(self, tokens, layer):
        # tokens : (B, L)
        # layer (1->n_layers): will stop the forward pass just after this layer

        # x : (B, L, D) activations after {layer}

        x = self.embedding(tokens)
        x = self.core(x, stop_at_layer=layer)

        return x
    
    # taken from llama2.c
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # taken from llama2.c
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # any parameters that is 2D will be weight decayed, otherwise no. (i.e. all weight tensors in matmuls + embeddings decay, all biases and rmnsnorms don't)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
