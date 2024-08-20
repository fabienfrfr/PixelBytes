#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import cv2, re, unicodedata
import requests, pandas as pd
from bs4 import BeautifulSoup
import numpy as np, pylab as plt
from numpy.lib.stride_tricks import as_strided
from skimage import color

# "Tokenizer"
Pixelbytes_tokens =  [
    ## Bytes (ASCII - UTF8)
    b'\x00', b'\t', b'\n', b' ', b'"', b"'", b'(', b')', b'*', b',', b'-', b'.', 
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'\xc2', b'\xa0', 
    b':', b'[', b']', b';', b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', 
    b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', 
    b'w', b'x', b'y', b'z',
    ## Pixel (RGB NES Palette)
    [0x00, 0x00, 0x00], [0xfc, 0xfc, 0xfc], [0xf8, 0xf8, 0xf8], [0xbc, 0xbc, 0xbc],
    [0x7c, 0x7c, 0x7c], [0xa4, 0xe4, 0xfc], [0x3c, 0xbc, 0xfc], [0x00, 0x78, 0xf8],
    [0x00, 0x00, 0xfc], [0xb8, 0xb8, 0xf8], [0x68, 0x88, 0xfc], [0x00, 0x58, 0xf8],
    [0x00, 0x00, 0xbc], [0xd8, 0xb8, 0xf8], [0x98, 0x78, 0xf8], [0x68, 0x44, 0xfc],
    [0x44, 0x28, 0xbc], [0xf8, 0xb8, 0xf8], [0xf8, 0x78, 0xf8], [0xd8, 0x00, 0xcc],
    [0x94, 0x00, 0x84], [0xf8, 0xa4, 0xc0], [0xf8, 0x58, 0x98], [0xe4, 0x00, 0x58],
    [0xa8, 0x00, 0x20], [0xf0, 0xd0, 0xb0], [0xf8, 0x78, 0x58], [0xf8, 0x38, 0x00],
    [0xa8, 0x10, 0x00], [0xfc, 0xe0, 0xa8], [0xfc, 0xa0, 0x44], [0xe4, 0x5c, 0x10],
    [0x88, 0x14, 0x00], [0xf8, 0xd8, 0x78], [0xf8, 0xb8, 0x00], [0xac, 0x7c, 0x00],
    [0x50, 0x30, 0x00], [0xd8, 0xf8, 0x78], [0xb8, 0xf8, 0x18], [0x00, 0xb8, 0x00],
    [0x00, 0x78, 0x00], [0xb8, 0xf8, 0xb8], [0x58, 0xd8, 0x54], [0x00, 0xa8, 0x00],
    [0x00, 0x68, 0x00], [0xb8, 0xf8, 0xd8], [0x58, 0xf8, 0x98], [0x00, 0xa8, 0x44],
    [0x00, 0x58, 0x00], [0x00, 0xfc, 0xfc], [0x00, 0xe8, 0xd8], [0x00, 0x88, 0x88],
    [0x00, 0x40, 0x58], [0xf8, 0xd8, 0xf8], [0x78, 0x78, 0x78]]

def input_seq_construct(arr, dim=3, sep_val=255):
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
        separator[:, 0, 1, 1] = sep_val
        result = np.concatenate((result, separator), axis=1)
        # Linearize
        return result.reshape(-1, dim, dim)[:-1][None]
    else :
        return result.reshape(-1, dim, dim)[None]

def image_pixelization(img, palette, max_size=25): 
    num_colors = len(palette)
    ## Cropping part
    # Binarize image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Find Rectangle
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Crop image
    x_min, y_min = max(0, x), max(0, y)
    x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)
    img = img[y_min:y_max, x_min:x_max]
    ## Pretreatment
    # Remove alpha (if)
    if img.shape[2] == 4:
        rgb_channels = img[:, :, :3]
        alpha_channel = img[:, :, 3] / 255.0
        # Merge image with white background
        white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        img = cv2.convertScaleAbs(rgb_channels * alpha_channel[..., None] + white_background * (1 - alpha_channel[..., None]))
    # Scale factor calculation
    h, w = img.shape[:2]
    scale_factor = max_size / max(h, w)
    # resizing
    new_height = int(min(h,w) * scale_factor)
    img = cv2.resize(img, (2*new_height, 2*max_size) if h>w else (2*max_size, 2*new_height), interpolation=cv2.INTER_NEAREST)
    ## Quantize image
    # Flatten image and convertion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    # K-Means Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Change pixel following cluster
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    img = cv2.cvtColor(quantized.reshape(img.shape), cv2.COLOR_Lab2BGR)
    # resizing
    new_height = int(min(h,w) * scale_factor)
    img = cv2.resize(img, (new_height, max_size) if h>w else (max_size, new_height), interpolation=cv2.INTER_NEAREST)
    ## Palette association
    palette_rgb = palette.copy()
    palette = color.rgb2lab(palette_rgb.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    def closest_color(pixel, palette):
        pixel_lab = color.rgb2lab(pixel.reshape(1, 1, 3) / 255.0).reshape(3)
        # distance
        distances = np.linalg.norm(palette - pixel_lab, axis=1)
        # Retourner la couleur la plus proche en RGB
        return palette_rgb[np.argmin(distances)]
    img = np.apply_along_axis(closest_color, 2 , img, palette)
    return img