#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import torch
from torch import nn
from dataclasses import dataclass
from mamba_ssm.modules.mamba_simple import Mamba

@dataclass
class MambaConfig:
    dim: int # The input dimension of the input tensor.
    d_state: int = 16 #16 # The dimension of the state space model.
    d_conv : int = 4 # The convolutionnal windows
    expand: int = 2 # E in paper/comments
    depth : int = 8 # The number of residual S6 layers
    vocab_size : int = 256+216+2 # ASCII bytes + RGB 6*6*6 pixel + img step/stop

class BysMamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        # text & image(t) embedding
        self.linear_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.patch_embedding = nn.Conv2d(in_channels=config.dim, out_channels=config.dim, kernel_size=3, stride=1, padding=0) # 3D in future
        # mamba part
        self.in_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        self.layers = nn.ModuleList([Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,) for _ in range(config.depth)])
        self.out_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        # output
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x):
        # shape : x : (B, L, M=3, N=3) : long
        B,L,M,N = x.shape
        dim = self.config.dim
        # embedding
        x = self.linear_embedding(x.view(B*L, M, N)).squeeze()
        x = x.permute(0, 3, 1, 2)  # (batch_size*L, embedding_dim, height, width)
        x = ((x[:,:,M//2,N//2].squeeze() + self.patch_embedding(x)).view(B, L, dim))/2 # (B,L,D)
        # bidirectional mamba input
        x += (self.in_mamba(x) + self.in_mamba(torch.flip(x, dims=[1])).flip([1]))/2
        # mamba intermediate layers
        for layer in self.layers:
            x += layer(x)
        # bidirectional mamba output
        x += (self.out_mamba(x) + self.out_mamba(torch.flip(x, dims=[1])).flip([1]))/2
        # prediction output
        x = self.lm_head(x) # probability
        return x