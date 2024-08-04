#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

adapted and simplified from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from .pscan import pscan

class SSM(nn.Module):
    def __init__(self, d_inner, dt_rank, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x):
        ## Runs the SSM
        # Compute ∆ A B C D, the state space parameters.
        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D) # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED) y : (B, L, ED) Δ : (B, L, ED)
        # A : (ED, N) B : (B, L, N)  C : (B, L, N) D : (ED)
        
        _, L, _ = x.shape
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        # parallel or seq
        if torch.cuda.is_available() :
            hs = pscan(deltaA, BX)
        else :
            h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device) # (B, ED, N)
            hs = [deltaA[:, t] * h + BX[:, t] for t in range(0, L)]
            hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x
        return y
