#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import torch, os
from torch import nn
#import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoConfig, AutoModel

try :
    from mamba_ssm.modules.mamba_simple import Mamba
except :
    print('No mamba_ssm modules installed...')

### Multimodal embedding        
class PxByEmbed(nn.Module):
    def __init__(self, vocab_size, dim, is_pemb=True, k=3):
        super().__init__()
        self.d_model, self.k = dim, k
        self.e_model = max(8, dim // (k**2))
        self.is_p_embed = is_pemb
        # Spatially adaptive embedding combination
        self.alpha = nn.Parameter(torch.rand(1, 1, k, k))
        self.projection = nn.Linear(self.e_model * k * k, dim)
        # Final normalization
        self.norm = nn.LayerNorm(dim)
        # Classic text embedding
        self.linear_embedding = nn.Embedding(vocab_size, self.e_model)
        # Local embedding patch
        self.patch_embedding = nn.Conv2d(in_channels=self.e_model, out_channels=self.e_model, 
                                         kernel_size=k, stride=1, padding=k//2) if is_pemb else None
    def forward(self, x):
        # shape : x : (B, L, M=k, N=k) : long
        B, L, M, N = x.shape
        assert M == N == self.k, f"Input spatial dimensions should be {self.k}x{self.k}, but got {M}x{N}"
        # Linear embedding
        x = self.linear_embedding(x.view(B*L, M, N))  # (B*L, M, N, e_model)
        x = x.permute(0, 3, 1, 2)  # (B*L, e_model, M, N)
        # Combine linear (and patch) embedding with alpha
        if not(self.is_p_embed): x = x = torch.sigmoid(self.alpha) * x  # (B*L, e_model, M, N)
        else: x = torch.sigmoid(self.alpha) * x + (1 - torch.sigmoid(self.alpha)) * self.patch_embedding(x) 
        # Flatten and project
        x = x.reshape(B*L, -1)  # (B*L, e_model*M*N)
        x = self.projection(x)  # (B*L, dim)
        # Final normalization
        x = self.norm(x)
        return x.view(B, L, -1)  # (B, L, dim)

### Main model
@dataclass
class ModelConfig:
    dim : int # The input dimension of the input tensor. (embedding dim output)
    pembed : bool = True # convolutionnal embedding
    bidirectional : bool = True # For RNN or SSM model
    d_state: int = 16 #16 # The dimension of the state space model.
    d_conv : int = 4 # The convolutionnal windows
    expand: int = 2 # E in paper/comments
    depth : int = 3 # The number of residual layers
    vocab_size : int = 113 # ASCII bytes + NES Pixel

class bMamba(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.bidirectional = config.bidirectional
        self.pxby_embedding = PxByEmbed(config.vocab_size, config.dim, config.pembed)
        # First Mamba layer (bidirectional or not based on config)
        self._mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
        # Remaining Mamba layers
        self.layers = nn.ModuleList([
            Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
            for _ in range(config.depth - 1)
        ]) if config.depth > 1 else None
        self.norm = nn.LayerNorm(config.dim) if config.bidirectional else None
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x):
        x = self.pxby_embedding(x)
        # Bidirectional Mamba for the first layer
        if self.bidirectional: x = self.norm(self._mamba(x) + self._mamba(torch.flip(x, dims=[1])).flip([1]))
        else: x = self._mamba(x)
        # Remaining Mamba layers
        if self.layers:
            for layer in self.layers: x = layer(x)
        return self.lm_head(x)

### Comparizon model
# simple lstm (like simplified PixelRNN)
class SimpleRNNModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimpleRNNModel, self).__init__()
        self.embedding = PxByEmbed(config.vocab_size, config.dim, config.pembed)
        # First LSTM layer (bidirectional or not based on config)
        self._lstm = nn.LSTM(input_size=config.dim,
            hidden_size=config.d_state // (2 if config.bidirectional else 1),
            num_layers=1,batch_first=True,bidirectional=config.bidirectional)
        # Remaining LSTM layers (if any)
        input_size = config.d_state if config.bidirectional else config.d_state // 2
        self.lstm = nn.LSTM( input_size=input_size,hidden_size=config.d_state, 
            num_layers=config.depth - 1, batch_first=True) if config.depth > 1 else None
        # Fully connected layer
        self.fc = nn.Linear(config.d_state, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self._lstm(x)[0]
        x = x if self.lstm is None else self.lstm(x)[0]
        return self.fc(x[:, -1, :])

# simple attention (like VERY simplified GPeT)
class SimpleTransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimpleTransformerModel, self).__init__()
        self.num_heads = 4  # to adjust
        self.num_layers = config.depth
        self.vocab = config.vocab_size
        self.embedding_dim = (config.dim // self.num_heads) * self.num_heads
        # Embedding
        self.embedding = PxByEmbed(config.vocab_size, self.embedding_dim, config.pembed)
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
        return self.fc(x[:, -1, :])

### convert model to HF (to push to hub)
def convert_pytorch_to_HF(source_dir, target_dir, model_type="bert"):
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith('.pth'):
            state_dict = torch.load(os.path.join(source_dir, filename), map_location='cpu')
            config = AutoConfig.from_pretrained(model_type)
            model = AutoModel.from_config(config)
            model.load_state_dict(state_dict, strict=False)
            target_model_dir = os.path.join(target_dir, filename[:-4])
            model.save_pretrained(target_model_dir)
            config.save_pretrained(target_model_dir)
            print(f"Model saved : {target_model_dir}")


