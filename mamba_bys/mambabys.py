#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

(IN DEV)
"""

import torch, math
import skimage as sk
import numpy as np, pylab as plt
from torch import nn
from numpy.lib.stride_tricks import as_strided
from dataclasses import dataclass
from skimage import io, transform, img_as_ubyte
from mamba_ssm.modules.mamba_simple import Mamba

def Color_rquantization(img, max_size=20, nb_colors=216):
    # Scale factor calculation
    m, n, _ = img.shape
    scale_factor = max_size / max(m,n) # image.shape[1] # to update to calculate max beetween L or l
    # Resize image
    new_height = int(min(m,n) * scale_factor) #     new_height = int(image.shape[0] * scale_factor)
    if n>m : resized_image  = transform.resize(img, (new_height, max_size), anti_aliasing=True)
    else : resized_image  = transform.resize(img, (max_size, new_height), anti_aliasing=True)
    ## Construct palette
    img = img_as_ubyte(resized_image)
    # cubic root numerotation 
    c = np.rint(np.cbrt(nb_colors)).astype(int)
    # reduce to x level per channel
    img_x = np.rint(img/(255/(c-1))).astype(np.uint8)
    # Convertir image 0 to nb_colors
    img_c = img_x[:, :, 0] * c**2 + img_x[:, :, 1] * c + img_x[:, :, 2]
    # Convertir en image 8 bits
    return img_c.astype(np.uint8)

def input_seq_construct(arr, dim=3):
    if dim % 2 == 0 : dim +=1
    pw = (dim-1)//2
    # Padding around array and size
    padded_arr = np.pad(arr, pad_width=pw, mode='constant', constant_values=0)
    shape = (arr.shape[0], arr.shape[1], dim, dim)
    strides = padded_arr.strides * 2  # double step
    # Data matrix construct
    matrix_dxd = as_strided(padded_arr, shape=shape, strides=strides)
    # include time asymetry (need correction for dim!=3)
    result = np.copy(matrix_dxd).astype(np.uint8)
    result[:, :, pw:, -pw:] = 0
    result[:, :, -pw, :] = 0
    # if image data, including skip line
    if shape[0] > 1 :
        separator = np.zeros((shape[0], 1, dim, dim), dtype=np.uint8)
        separator[:, 0, 1, 1] = 255
        result = np.concatenate((result, separator), axis=1)
        # Linearize
        return result.reshape(-1, dim, dim)[:-1]
    else :
        return result.reshape(-1, dim, dim)

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
        self.vocab_size = 256+216 # ASCII bytes + RGB 6*6*6 pixel
        self.linear_embedding = nn.Embedding(self.vocab_size, config.dim)
        self.patch_embedding = nn.Conv2d(in_channels=1, out_channels=self.vocab_size, kernel_size=3, stride=1, padding=0) # 3D in future
        # mamba part
        self.in_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        self.layers = nn.ModuleList([Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,) for _ in range(config.depth)])
        self.out_mamba = Mamba(d_model=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,)
        # output
        self.lm_head = nn.Linear(config.dim, self.vocab_size, bias=False)

    def forward(self, x):
        # shape : x : (B, L, M=3, N=3)
        B,L,M,N = x.shape
        # embedding
        #xl = x[:, :, M//2, N//2] # img center
        #xl = self.linear_embedding(xl) # (B,L,D)
        xp = self.patch_embedding(input_tensor.view(B*N, 1, 3, 3)).view(B, L, self.vocab_size)
        x = xp #+ xl
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
