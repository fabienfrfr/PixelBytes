#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

try :
    from mamba_ssm.modules.mamba_simple import Mamba
except :
    print('No mamba_ssm modules installed...')

### Multimodal embedding        
class PxByEmbed(nn.Module):
    def __init__(self, vocab_size, dim, k=3):
        super().__init__()
        self.d_model, self.k = dim, k
        self.e_model = max(8, dim // (k**2))
        # Spatially adaptive embedding combination
        self.alpha = nn.Parameter(torch.rand(1, 1, k, k))
        self.projection = nn.Linear(self.e_model * k * k, dim)
        # Final normalization
        self.norm = nn.LayerNorm(dim)
        # Classic text embedding
        self.linear_embedding = nn.Embedding(vocab_size, self.e_model)
        # Local embedding patch
        self.patch_embedding = nn.Conv2d(in_channels=self.e_model, out_channels=self.e_model, 
                                         kernel_size=k, stride=1, padding=k//2)
    def forward(self, x):
        # shape : x : (B, L, M=k, N=k) : long
        B, L, M, N = x.shape
        assert M == N == self.k, f"Input spatial dimensions should be {self.k}x{self.k}, but got {M}x{N}"
        # Linear embedding
        x = self.linear_embedding(x.view(B*L, M, N))  # (B*L, M, N, e_model)
        x = x.permute(0, 3, 1, 2)  # (B*L, e_model, M, N)
        # Combine linear and patch embedding with spatial alpha
        patch_emb = self.patch_embedding(x)
        x = torch.sigmoid(self.alpha) * x + (1 - torch.sigmoid(self.alpha)) * patch_emb  # (B*L, e_model, M, N)
        # Flatten and project
        x = x.reshape(B*L, -1)  # (B*L, e_model*M*N)
        x = self.projection(x)  # (B*L, dim)
        # Final normalization
        x = self.norm(x)
        return x.view(B, L, -1)  # (B, L, dim)

### Main model
@dataclass
class ModelConfig:
    dim: int # The input dimension of the input tensor. (embedding dim output)
    d_state: int = 16 #16 # The dimension of the state space model.
    d_conv : int = 4 # The convolutionnal windows
    expand: int = 2 # E in paper/comments
    depth : int = 3 # The number of residual layers
    vocab_size : int = 113 # ASCII bytes + NES Pixel

class bMamba(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # text & image(t) embedding
        self.pxby_embedding = PxByEmbed(config.vocab_size, config.dim)
        # mamba part
        self._mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        self.layers = nn.ModuleList([Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,) 
                                     for _ in range(max(1,config.depth-1))])
        # norm & output
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x):
        # pixelbyte embedding
        x = self.pxby_embedding(x)
        # bidirectional mamba input & norm
        x = self._mamba(x) + self._mamba(torch.flip(x, dims=[1])).flip([1])
        x = self.norm(x)
        # mamba intermediate layers
        for layer in self.layers:
            x = layer(x)
        # prediction output
        x = self.lm_head(x) # probability
        return x

### Comparizon model
# simple lstm (like simplified PixelRNN)
class SimpleRNNModel(nn.Module):
    def __init__(self, config: ModelConfig): #vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNNModel, self).__init__()
        self.embedding_dim = config.dim
        self.hidden_dim = config.d_state
        self.nlayer = config.depth
        self.vocab = config.vocab_size
        # model
        self.embedding = PxByEmbed(self.vocab, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.nlayer, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.vocab)

    def forward(self, x):
        # x shape: [batch_size, seq_length, 3, 3]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Use only the last output
        return output

# simple attention (like VERY simplified GPeT)
class SimpleTransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimpleTransformerModel, self).__init__()
        self.num_heads = 4  # to adjust
        self.num_layers = config.depth
        self.vocab = config.vocab_size
        self.embedding_dim = (config.dim // self.num_heads) * self.num_heads
        # Embedding
        self.embedding = PxByEmbed(self.vocab, self.embedding_dim)
        # Transformer 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, 
                                                   nhead=self.num_heads,
                                                   dim_feedforward=4 * self.embedding_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=self.num_layers)
        # Output layer
        self.fc = nn.Linear(self.embedding_dim, self.vocab)

    def forward(self, x):
        # x shape: [batch_size, seq_length, 3, 3]
        x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        # Use the last token's representation for classification
        output = self.fc(x[:, -1, :])
        return output