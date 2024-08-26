#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import torch, re
import torch.nn as nn
from huggingface_hub import HfApi, create_repo, whoami
from transformers import PreTrainedModel, PretrainedConfig
from dataclasses import dataclass

try :
    from mamba_ssm.modules.mamba_simple import Mamba
except :
    print('No mamba_ssm modules installed...')

### Multimodal embedding        
class PxByEmbed(nn.Module):
    def __init__(self, vocab_size, dim, is_pemb=True, k=3):
        super().__init__()
        self.d_model, self.k = dim, k
        self.e_model = max(9, dim // (k**2))
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
class ModelConfig(PretrainedConfig):
    dim : int # The input dimension of the input tensor. (embedding dim output)
    model_type: str = "sequence-generator"
    pembed : bool = True # convolutionnal embedding
    pxbx_embed : bool = True # PixelBytes or only center sequence
    bidirectional : bool = True # For RNN or SSM model
    d_state: int = 16 # The dimension of the state space model
    d_conv : int = 4 # The convolutionnal windows
    expand: int = 2 # E in paper/comments
    depth : int = 3 # The number of residual layers
    vocab_size : int = 113 # ASCII bytes + NES Pixel
    
    def __init__(self, dim: int = None, **kwargs):
        super().__init__(**kwargs)
        if dim is not None:
            self.dim = dim
        
class bMamba(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = "simple_ssm"
    
    def __init__(self, config: ModelConfig):
        super(bMamba, self).__init__(config)
        self.name = "ssm"
        self.bidirectional, self.pxby = config.bidirectional, config.pxbx_embed
        self.embedding = PxByEmbed(config.vocab_size, config.dim, config.pembed) if self.pxby else nn.Embedding(config.vocab_size, config.dim)
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
        x = self.embedding(x) if self.pxby else self.embedding(x[:,:,1,1])
        # Bidirectional Mamba for the first layer
        if self.bidirectional: x = self.norm(self._mamba(x) + self._mamba(torch.flip(x, dims=[1])).flip([1]))
        else: x = self._mamba(x)
        # Remaining Mamba layers
        if self.layers:
            for layer in self.layers: x = layer(x)
        return self.lm_head(x)

### Comparizon model
# simple lstm (like simplified PixelRNN)
class SimpleRNNModel(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = "simple_rnn"
    
    def __init__(self, config: ModelConfig):
        super(SimpleRNNModel, self).__init__(config)
        self.name = "rnn"
        self.pxby = config.pxbx_embed
        self.embedding = PxByEmbed(config.vocab_size, config.dim, config.pembed) if self.pxby else nn.Embedding(config.vocab_size, config.dim)
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
        x = self.embedding(x) if self.pxby else self.embedding(x[:,:,1,1])
        x = self._lstm(x)[0]
        x = x if self.lstm is None else self.lstm(x)[0]
        return self.fc(x[:, -1, :])

# simple attention (like VERY simplified GPeT)
class SimpleTransformerModel(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = "simple_transformer"
    
    def __init__(self, config: ModelConfig):
        super(SimpleTransformerModel, self).__init__(config)
        self.name = "attention"
        self.pxby = config.pxbx_embed
        self.num_heads = 4  # to adjust
        self.num_layers = config.depth
        self.vocab = config.vocab_size
        self.embedding_dim = (config.dim // self.num_heads) * self.num_heads
        # Embedding
        self.embedding = PxByEmbed(config.vocab_size, self.embedding_dim, config.pembed) if self.pxby else nn.Embedding(config.vocab_size, self.embedding_dim)
        # Transformer 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, 
                                                   nhead=self.num_heads,
                                                   dim_feedforward=4 * self.embedding_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=self.num_layers)
        # Output layer
        self.fc = nn.Linear(self.embedding_dim, self.vocab)

    def forward(self, x):
        # x shape: [batch_size, seq_length, 3, 3]
        x = self.embedding(x) if self.pxby else self.embedding(x[:,:,1,1])
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        # Use the last token's representation for classification
        return self.fc(x[:, -1, :])


### save model function
def push_model_to_hub(repo_name, model_dir, token, subfolder=None):
    api = HfApi(token=token)
    subfolder = re.sub(r'[^a-zA-Z0-9]+', '_', subfolder).strip('_').lower()

    try:
        create_repo(repo_name, token=token, repo_type="model", exist_ok=True)
        username = whoami(token=token)['name']
        repo_id = f"{username}/{repo_name}"
        print(f"Repository '{repo_id}' created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=subfolder,
        ignore_patterns=[".*"],  # Ignorer les fichiers cachés
        create_pr=False  # Créer directement dans la branche principale
    )
    print(f"Model pushed successfully to {repo_name}, subfolder: {subfolder}")