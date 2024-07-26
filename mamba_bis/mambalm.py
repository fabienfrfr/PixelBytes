#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

from .mambabis import MambaConfig, BiMamba

import torch, math
import inspect
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from einops.layers.torch import Reduce
from dataclasses import dataclass

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class BiMambaLM(nn.Module):
    def __init__(self, model_config: MambaConfig, vocab_size: int):
        super().__init__()
        
        self.config = model_config
        # model
        self.embedding = nn.Embedding(vocab_size, self.config.dim, padding_idx=0)
        self.core = BiMamba(self.config)
        self.out_norm = RMSNorm(self.config.dim, self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.dim, vocab_size, bias=False)
        
        # share & init normal weight
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.depth))
        
    def forward(self, tokens):
        # tokens : (B, L) # logits : (B, L, vocab_size)
        x = self.embedding(tokens)
        x = self.core(x)
        x = self.out_norm(x)
        logits = self.lm_head(x)
        return logits

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
