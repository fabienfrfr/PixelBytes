#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

from transformers import PreTrainedTokenizer
import numpy as np, os, unicodedata
from skimage.color import rgb2lab
from typing import List, Dict, Union, Tuple
from scipy.spatial.distance import cdist

## Bytes (ASCII - UTF8)
DEFAULT_BYTES = [b'\x00', b'\t', b'\n', b' ', b'"', b"'", b'(', b')', b'*', b',', b'-', b'+', 
                b'.', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'\xc2', 
                b'\xa0', b':', b'[', b']', b';', b'/', b'%', b'!', b'a', b'b', b'c', b'd', b'e', 
                b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', 
                b's', b't', b'u', b'v', b'w', b'x', b'y', b'z']
## Pixel (RGB NES Palette)
DEFAULT_PALETTE = [(0x00, 0x00, 0x00), (0xfc, 0xfc, 0xfc), (0xf8, 0xf8, 0xf8), (0xbc, 0xbc, 0xbc),
                    (0x7c, 0x7c, 0x7c), (0xa4, 0xe4, 0xfc), (0x3c, 0xbc, 0xfc), (0x00, 0x78, 0xf8),
                    (0x00, 0x00, 0xfc), (0xb8, 0xb8, 0xf8), (0x68, 0x88, 0xfc), (0x00, 0x58, 0xf8),
                    (0x00, 0x00, 0xbc), (0xd8, 0xb8, 0xf8), (0x98, 0x78, 0xf8), (0x68, 0x44, 0xfc),
                    (0x44, 0x28, 0xbc), (0xf8, 0xb8, 0xf8), (0xf8, 0x78, 0xf8), (0xd8, 0x00, 0xcc),
                    (0x94, 0x00, 0x84), (0xf8, 0xa4, 0xc0), (0xf8, 0x58, 0x98), (0xe4, 0x00, 0x58),
                    (0xa8, 0x00, 0x20), (0xf0, 0xd0, 0xb0), (0xf8, 0x78, 0x58), (0xf8, 0x38, 0x00),
                    (0xa8, 0x10, 0x00), (0xfc, 0xe0, 0xa8), (0xfc, 0xa0, 0x44), (0xe4, 0x5c, 0x10),
                    (0x88, 0x14, 0x00), (0xf8, 0xd8, 0x78), (0xf8, 0xb8, 0x00), (0xac, 0x7c, 0x00),
                    (0x50, 0x30, 0x00), (0xd8, 0xf8, 0x78), (0xb8, 0xf8, 0x18), (0x00, 0xb8, 0x00),
                    (0x00, 0x78, 0x00), (0xb8, 0xf8, 0xb8), (0x58, 0xd8, 0x54), (0x00, 0xa8, 0x00),
                    (0x00, 0x68, 0x00), (0xb8, 0xf8, 0xd8), (0x58, 0xf8, 0x98), (0x00, 0xa8, 0x44),
                    (0x00, 0x58, 0x00), (0x00, 0xfc, 0xfc), (0x00, 0xe8, 0xd8), (0x00, 0x88, 0x88),
                    (0x00, 0x40, 0x58), (0xf8, 0xd8, 0xf8), (0x78, 0x78, 0x78)]
## Action-space (Control & Audio)
DEFAULT_ACTION_STATE = np.linspace(-1, 1, 11).tolist()

##### Tokenizer
class ActionPixelBytesTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        ## Bytes (ASCII - UTF8) + Pixel (RGB NES Palette) + Action-space (Control & Audio)
        ActionPixelbytes =  DEFAULT_BYTES + DEFAULT_PALETTE + DEFAULT_ACTION_STATE
        self.vocab = {ActionPixelbytes[i] : i for i in range(len(ActionPixelbytes))}
        super().__init__(**kwargs)
        self.bytes_size = len(DEFAULT_BYTES) # first is null values
        self.palet_size = len(DEFAULT_PALETTE)
        self.LabPalette = rgb2lab(np.array(DEFAULT_PALETTE)[None, :, :] / 255.0)[0]
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return {k: v for k, v in self.vocab.items()}

    def _convert_token_to_id(self, token: Union[bytes, tuple]) -> int:
        return self.vocab.get(token, self.vocab.get(b'[UNK]', 0))

    def _convert_id_to_token(self, index: int) -> Union[bytes, tuple]:
        return self.ids_to_tokens.get(index, b'[UNK]')

    def convert_tokens_to_ids(self, tokens: List[Union[bytes, tuple]]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]
    
    def decode(self, token_ids, skip_special_tokens=True):
        return [self._convert_id_to_token(id) for id in token_ids]

    def __call__(self, text=None, image=None, action_state=None, **kwargs):
        inputs = []
        if text is not None:
            inputs.append(self.process_text(text))
        if image is not None:
            inputs.append(self.process_image(image))
        if action_state is not None:
            inputs.append(self.process_action_state(action_state))
        if not inputs:
            raise ValueError("At least one input (text, image, or action_state) must be provided")
        context, targets = zip(*[self.create_sequence_data(inp) for inp in inputs])
        return { "input_ids": np.concatenate(context),
                    "labels": np.concatenate(targets)}

    def process_text(self, text):
        text = [bytes([b]) for b in unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore')]
        return np.array(self.convert_tokens_to_ids(text))[None,None,:]

    def process_image(self, image):
        n_frames = getattr(image, "n_frames", 1) # PIL Image (GIF compatible)
        frames_array = np.empty((n_frames, image.height, image.width), dtype=np.uint8)
        for i in range(n_frames):
            image.seek(i)
            frame_lab = rgb2lab(np.array(image.convert('RGB')) / 255.0)
            frames_array[i] = cdist(frame_lab.reshape(-1, 3), self.LabPalette).argmin(axis=1).reshape(image.size[::-1])
        return frames_array + self.bytes_size # Tips (pass all bytes ids)

    def process_action_state(self, action_state):
        # normalization (T,2) 0 : Action; 1 : State (Dataset BangBang control for State = 0 : see GymSetpoint
        normalized_state = action_state / np.linalg.norm(action_state, axis=1, keepdims=True)
        indices = cdist(normalized_state, DEFAULT_ACTION_STATE).argmin(axis=1)
        return indices[:, None, None] + self.bytes_size + self.palet_size

    def create_sequence_data(self, context_array):
        n_frames, height, width = context_array.shape
        padded = np.pad(context_array, ((1, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        slices = [  padded[:-1, 1:-1, 1:-1],  # (t-1, i, j)
                    padded[:-1, 1:-1, 2:],    # (t-1, i, j+1)
                    padded[:-1, :-2, 1:-1],   # (t-1, i-1, j)
                    padded[1:, :-2, :-2],     # (t, i-1, j-1)
                    padded[1:, 1:-1, :-2],    # (t, i, j-1)
                    padded[1:, :-2, 1:-1]]   # (t, i-1, j)
        context = np.stack(slices, axis=-1)
        return context.reshape(-1, 6), context_array.reshape(-1, 1)

### basic test
if __name__ == '__main__' :
    tokenizer = ActionPixelBytesTokenizer()
    from datasets import load_dataset

    print("https://www.perplexity.ai/search/est-ce-qu-il-existe-des-tokeni-EXjFBpLKSje6npL0pYuU5Q")
    text = "Hello, world!"
    text_ids = tokenizer(text=text)
    print(text_ids)
    anim_dataset = load_dataset("ffurfaro/PixelBytes-PokemonSprites")
    bulbi_back = img = anim_dataset['train']['image'][0]
    img_ids = tokenizer(image=img)
    print(img_ids)
    pxby_dataset = load_dataset("ffurfaro/PixelBytes-Pokemon")
    bulbi = img = pxby_dataset['train']['image'][0]
    img_ids = tokenizer(image=img)
    print(img_ids)




