#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import torch
from torch import nn
from dataclasses import dataclass

try :
    from mamba_ssm.modules.mamba_simple import Mamba
except :
    print('No mamba_ssm modules installed...')

@dataclass
class MambaConfig:
    dim: int # The input dimension of the input tensor.
    d_state: int = 16 #16 # The dimension of the state space model.
    d_conv : int = 4 # The convolutionnal windows
    expand: int = 2 # E in paper/comments
    depth : int = 8 # The number of residual S6 layers
    vocab_size : int = 110 # ASCII bytes + NES Pixel

class PxByEmbed(nn.Module):
    def __init__(self, vocab_size, dim, k=3):
        super().__init__()
        self.d_model = dim
        # classic text embedding
        self.linear_embedding = nn.Embedding(vocab_size, dim)
        # local embedding patch (3D in future)
        self.patch_embedding = nn.Conv2d(in_channels=dim, out_channels=dim, 
                                         kernel_size=k, stride=1, padding=0)
    def forward(self, x):
        # shape : x : (B, L, M=3, N=3) : long
        B,L,M,N = x.shape
        dim = self.d_model
        # embedding
        x = self.linear_embedding(x.view(B*L, M, N)).squeeze()
        x = x.permute(0, 3, 1, 2)  # (batch_size*L, embedding_dim, height, width)
        x = ((x[:,:,M//2,N//2].squeeze() + self.patch_embedding(x)).view(B, L, dim))/2 # (B,L,D)
        return x

class BysMamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        # text & image(t) embedding
        self.pxby_embedding = PxByEmbed(config.vocab_size, config.dim)
        # mamba part
        self.in_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        self.layers = nn.ModuleList([Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,) for _ in range(config.depth)])
        self.out_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        # output
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x):
        ## shape : x : (B, L, M, N) : long
        # pixelbyte embedding
        x = self.pxby_embedding(x)
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

### Comparizon model
# simple lstm (like simplified PixelRNN)
class SimpleSeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleSeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 9, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, 3, 3]
        batch_size, seq_length, H, W = x.shape
        x = x.view(batch_size, seq_length, -1)  # Flatten 3x3 to 9
        embedded = self.embedding(x)  # [batch_size, seq_length, 9, embedding_dim]
        embedded = embedded.view(batch_size, seq_length, -1)  # Flatten embedding
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Use only the last output
        return output

# simple attention (like simplified GPT)
class SimpleAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads=1, num_layers=1):
        super(SimpleAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        attention_dim = ((embedding_dim * 9) // num_heads) * num_heads
        self.input_proj = nn.Linear(embedding_dim * 9, attention_dim)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers) ]) # Multilayers attention model
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim) 
            for _ in range(num_layers)])
        self.fc = nn.Linear(attention_dim, vocab_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, 3, 3]
        batch_size, seq_length, H, W = x.shape
        x = x.view(batch_size, seq_length, -1)  # Flatten 3x3 to 9
        embedded = self.embedding(x)  # [batch_size, seq_length, 9, embedding_dim]
        embedded = embedded.view(batch_size, seq_length, -1)  # Flatten to [batch_size, seq_length, 9*embedding_dim]
        embedded = self.input_proj(embedded)
        for i in range(len(self.attention_layers)):
            attn_output, _ = self.attention_layers[i](embedded, embedded, embedded)
            embedded = self.norm_layers[i](attn_output + embedded)  # residual
        output = self.fc(embedded[:, -1, :])  # last output
        return output

