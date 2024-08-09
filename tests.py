#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

TESTS FILE
"""

from mamba_bis.mambabis import MambaConfig, BiMamba
from mamba_bis.mambalm import BiMambaLM

import torch

### basic exemple (othello test)
if __name__ == '__main__' :
    # Model config
    vocab_size = 65 # Othello
    d_model = 256 # 288 for 
    n_layers = 8
    d_state = 16
    dim_inner = 256 #2 * d_state
    # MambaConfig(dim=256, dt_rank=32, d_inner=256, d_state=16, depth=8, expand_factor=2, d_conv=4, rms_norm_eps=1e-05) 
    # MambaPyConfig(d_model=256, n_layers=8, dt_rank=16, d_state=16, expand_factor=2, d_conv=4, rms_norm_eps=1e-05)
    config = MambaConfig(dim=d_model, depth=n_layers, d_state=d_state, d_inner=dim_inner)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fake input
    x = torch.randint(high=vocab_size, size=(256, 59))
    print(x.shape)
    
    # Initialize model
    model = BiMambaLM(config, vocab_size=vocab_size+7).to(device)
    # test
    logits = model(x) # (B, L, vocab_size)