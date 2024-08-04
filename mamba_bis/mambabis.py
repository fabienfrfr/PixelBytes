#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

adapted and simplified from https://github.com/alxndrTL/mamba.py/blob/main/mambapy/vim.py
"""

import torch, math
from einops import rearrange
from torch import nn, Tensor
from .ssm import SSM
from dataclasses import dataclass

@dataclass
class MambaConfig:
    dim: int # The input dimension of the input tensor.
    dt_rank: int = 'auto' # The rank of the state space model.
    d_inner: int = None # The dimension of the inner layer of the multi-head attention.
    d_state: int = None #16 # The dimension of the state space model.
    depth : int = 8 # The number of residual S6 layers
    expand_factor: int = 2 # E in paper/comments
    d_conv : int = 4 # The convolutionnal windows
    rms_norm_eps: float = 1e-5 # Root-mean-square normalization per episode
    
    def __post_init__(self):
        self.d_inner = self.expand_factor * self.dim # E*D = ED in comments
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.dim / self.d_state)

class BiMambaBlock(nn.Module):
    """
    BiMambaBlock is a module that implements Bidirectional State Space Model
    in.shape == out.shape
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.dim = config.dim
        self.dt_rank = config.dt_rank
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        
        # Bi-S6
        self.shared_conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=self.d_conv, bias=True, groups=config.d_inner, padding=config.d_conv - 1)
        self.norm = nn.LayerNorm(self.dim) # Standardization RMS
        self.silu = nn.SiLU()
        self.ssm = SSM(self.d_inner, self.dt_rank, self.d_state) # Shared

        # Linear layer for z and x
        self.in_proj = nn.Linear(self.dim, 2 * self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=False)
        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        # Skip connection
        skip = x
        # Normalization
        x = self.norm(x)
        # Split x into x1 and x2 with linears
        z = self.in_proj(x)
        (x1, z1) = z.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        (x2, z2) = z.flip([1]).split(split_size=[self.d_inner, self.d_inner], dim=-1)
        # forward & backward con1d (shared)
        x1 = self.process_direction(x1,self.shared_conv1d,self.ssm, s)
        x2 = self.process_direction(x2,self.shared_conv1d,self.ssm, s)
        # Activation
        z1 = self.silu(z1)
        z2 = self.silu(z2)
        # Matmul
        x1 *= z1
        x2 *= z2
        # projection
        x = self.out_proj((x1+x2.flip([1])) / 2)
        # Residual connection
        return x + skip

    def process_direction(self,x: Tensor,conv1d: nn.Conv1d,ssm: SSM, s:int):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x)[:, :, :s])
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x

class BiMamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([BiMambaBlock(config) for _ in range(config.depth)])

    def forward(self, x):
        # x : (B, L, D) == y : (B, L, D)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def step(self, x, caches):
        # caches : [cache(layer) for all layers], cache : (h, inputs)
        # initial caches : [(None, torch.zeros(B, self.config.d_inner, self.config.d_conv-1)) for _ in range(self.config.depth)]
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches
