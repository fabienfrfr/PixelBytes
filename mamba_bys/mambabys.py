#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

(IN DEV)
"""

import torch, math
from torch import nn
from mamba_ssm.modules.mamba_simple import Mamba
from dataclasses import dataclass

@dataclass
class MambaConfig:
    dim: int # The input dimension of the input tensor.
    d_state: int = 16 #16 # The dimension of the state space model.
    d_conv : int = 4 # The convolutionnal windows
    expand: int = 2 # E in paper/comments
    depth : int = 8 # The number of residual S6 layers

class BysMamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        # text & image(t) embedding
        self.vocab_size = 256+512 # ASCII bytes + RGB 8*8*8 pixel
        self.linear_embedding = nn.Embedding(self.vocab_size, config.dim)
        self.patch_embedding = nn.Conv2d(1, self.vocab_size, kernel_size=patch_size, stride=stride) # 3D in future
        # mamba part
        self.in_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        self.layers = nn.ModuleList([Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,) for _ in range(config.depth)])
        self.out_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        # output
        self.lm_head = nn.Linear(config.dim, self.vocab_size, bias=False)

    def forward(self, x):
        # shape : x : (B, M, N, L)
        _,M,N,_ = x.shape
        # embedding
        xl = x[:, M//2, N//2, :] # img center
        xl = self.linear_embedding(xl) # (B,L,D)
        xp = self.patch_embedding(x).flatten(2).transpose(1, 2) # (B,L,D)
        x = xl + xp
        # bidirectional mamba input
        x += self.in_mamba(x) + self.in_mamba(torch.flip(x, dims=[1])).flip([1])
        # mamba intermediate layers
        for layer in self.layers:
            x += layer(x)
        # bidirectional mamba output
        x += self.out_mamba(x) + self.out_mamba(torch.flip(x, dims=[1])).flip([1])
        # prediction output
        x = self.lm_head(x) # probability
        return x
