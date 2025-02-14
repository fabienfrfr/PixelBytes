#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import torch, unicodedata, colorsys
from torch import nn
from transformers import PreTrainedTokenizer
from skimage.color import rgb2lab
from typing import List, Dict, Union, Tuple
import numpy as np, os, ast

from dotenv import load_dotenv
load_dotenv()

## Bytes (ASCII - UTF8) 
DEFAULT_BYTES = [b'\x00', b'\t', b'\n', b' ', b'"', b"'", b'(', b')', b'*', b',', b'-', b'+', b'.', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'\xc2', b'\xa0', b':', b'[', b']', b';', b'/', b'%', b'!', b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z']
def generate_bytes(num_ascii=256):
    return [bytes([i]) for i in range(num_ascii)]
## Pixel (RGB NES Palette) 
DEFAULT_PALETTE = [tuple(p) for p in [[0, 0, 0], [252, 252, 252], [248, 248, 248], [188, 188, 188], [124, 124, 124], [164, 228, 252], [60, 188, 252], [0, 120, 248], [0, 0, 252], [184, 184, 248], [104, 136, 252], [0, 88, 248], [0, 0, 188], [216, 184, 248], [152, 120, 248], [104, 68, 252], [68, 40, 188], [248, 184, 248], [248, 120, 248], [216, 0, 204], [148, 0, 132], [248, 164, 192], [248, 88, 152], [228, 0, 88], [168, 0, 32], [240, 208, 176], [248, 120, 88], [248, 56, 0], [168, 16, 0], [252, 224, 168], [252, 160, 68], [228, 92, 16], [136, 20, 0], [248, 216, 120], [248, 184, 0], [172, 124, 0], [80, 48, 0], [216, 248, 120], [184, 248, 24], [0, 184, 0], [0, 120, 0], [184, 248, 184], [88, 216, 84], [0, 168, 0], [0, 104, 0], [184, 248, 216], [88, 248, 152], [0, 168, 68], [0, 88, 0], [0, 252, 252], [0, 232, 216], [0, 136, 136], [0, 64, 88], [248, 216, 248], [120, 120, 120]]]
def generate_palette(num_colors=55):
    return [tuple(int(255*x) for x in colorsys.hsv_to_rgb(i/num_colors, 0.8, 0.9)) for i in range(num_colors)]
## Action-space (Control & Audio)
def generate_action_space(num_steps=38):
    return np.linspace(-1, 1, num_steps).tolist()
DEFAULT_ACTION_STATE = generate_action_space()

##### Tokenizer
class ActionPixelBytesTokenizer(PreTrainedTokenizer):
    def __init__(self, BYTES=DEFAULT_BYTES, PALETTE=DEFAULT_PALETTE, ACTION_STATE=DEFAULT_ACTION_STATE,
                 data_slicing={"image":1, "audio":1}, **kwargs):
        ## Bytes (ASCII - UTF8) + Pixel (RGB NES Palette) + Action-space (Control & Audio)
        self.BYTES, self.PALETTE, self.ACTION_STATE = BYTES, PALETTE, ACTION_STATE
        ActionPixelbytes = BYTES + PALETTE + ACTION_STATE
        self.vocab = {ActionPixelbytes[i]: i for i in range(len(ActionPixelbytes))}
        super().__init__(**kwargs)
        self.slicing = data_slicing
        self.bytes_size = len(BYTES)
        self.palet_size = len(PALETTE)
        self.LabPalette = torch.tensor(rgb2lab(np.array(PALETTE)[None, :, :] / 255.0)[0])
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _convert_token_to_id(self, token: Union[bytes, tuple]) -> int:
        return self.vocab.get(token, self.vocab.get(b'[UNK]', 0))

    def _convert_id_to_token(self, index: int) -> Union[bytes, tuple]:
        return self.ids_to_tokens.get(index, b'[UNK]')

    def convert_tokens_to_ids(self, tokens: List[Union[bytes, tuple]]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]
    
    def decode(self, token_ids, skip_special_tokens=True):
        if token_ids.dim() > 1 : token_ids = token_ids.view(-1,6)[:,-1] # last padded value
        token_ids = token_ids.tolist()
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        text_tokens, image_frames, current_frame, current_line, audio_tokens = [], [], [], [], []
        for token in tokens:
            if isinstance(token, bytes):
                if token == b'\t':
                    if current_line:
                        current_frame.append(np.array(current_line)) # shape W,3
                        current_line = []
                elif token == b'\n':
                    if current_frame:
                        image_frames.append(current_frame)
                        current_line, current_frame = [], []
                elif token in self.BYTES:
                    text_tokens.append(token)
            elif isinstance(token, tuple):
                current_line.append(token)
            elif isinstance(token, (int, float)):
                audio_tokens.append(token)
        if current_line:
            current_frame.append(np.array(current_line))
            image_frames.append(current_frame)
        return {'text': b''.join(text_tokens),
                'image': image_frames,
                'audio': audio_tokens}

    def __call__(self, text=None, image=None, audio=None, **kwargs):
        inputs = []
        if text is not None:
            inputs.append(self.process_text(text))
        if image is not None:
            inputs.append(self.process_image(image))
        if audio is not None:
            inputs.append(self.process_action_state(audio['array']))
        if not inputs:
            raise ValueError("At least one input (text, image, or audio) must be provided")
        context, targets = zip(*[self.create_sequence_data(inp) for inp in inputs])
        return {"input_ids": torch.cat(context), "labels": torch.cat(targets)}

    def process_text(self, text):
        text = [bytes([b]) for b in unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore')]
        return torch.tensor(self.convert_tokens_to_ids(text) + [2], dtype=torch.long).unsqueeze(0).unsqueeze(0)

    def process_image(self, image):
        n_frames = getattr(image, "n_frames", 1)
        frames = torch.empty((n_frames, image.height, image.width), dtype=torch.long)
        for i in range(n_frames):
            image.seek(i)
            frame = torch.tensor(rgb2lab(np.array(image.convert('RGB')) / 255.0))
            distances = torch.cdist(frame.reshape(-1, 3), self.LabPalette)
            frames[i] = distances.argmin(dim=1).reshape(image.size[::-1])
        frames = (frames + self.bytes_size)[::self.slicing['image'], ::self.slicing['image'], ::self.slicing['image']]
        added_column = torch.ones((frames.shape[0], frames.shape[1], 1), dtype=torch.long)
        added_column[:, -1, :] = 2
        return torch.cat((frames, added_column), dim=2)

    def process_action_state(self, audio):
        audio = torch.tensor(audio, dtype=torch.float32).contiguous()
        if audio.dim() < 2: audio = audio.unsqueeze(0)
        normalized_state = ((audio - audio.min()) / (audio.max() - audio.min()) * 2 - 1).to(torch.float32)
        action_state_tensor = torch.tensor(self.ACTION_STATE, dtype=torch.float32).reshape(-1, 1)
        indices = torch.argmin(torch.cdist(normalized_state.view(-1, 1), action_state_tensor), dim=1).view(normalized_state.shape)
        indices = (indices + self.bytes_size + self.palet_size)[:, ::self.slicing['audio']]
        separators = torch.tensor([[1], [2]], dtype=torch.long)[:indices.shape[0]]
        return torch.cat([indices.long(), separators], dim=1).T.unsqueeze(1)

    def create_sequence_data(self, context_array):
        n_frames, height, width = context_array.shape
        padded = nn.functional.pad(context_array, (1, 1, 1, 1, 1, 0), mode='constant', value=0)
        slices = [
            padded[:-1, :-2, 1:-1], # (t-1, i-1, j) : up value in t-1
            padded[:-1, 1:-1, 1:-1],# (t-1, i, j) : value in t-1
            padded[:-1, 2:, 1:-1],  # (t-1, i+1, j) : down value in t-1
            padded[1:, :-2, :-2],   # (t, i-1, j-1) : up-left value in t
            padded[1:, :-2, 1:-1],  # (t, i-1, j) : up value in t
            padded[1:, 1:-1, :-2]]  # (t, i, j-1) : wrong previous value
        context = torch.stack(slices, dim=-1).reshape(-1, 6)
        targets = context_array.reshape(-1, 1)
        context[1:,-1] = targets[:-1,0] # true previous value (AutoRegressive)
        return context, targets

    def construct_images(self, images_token):
        img_list = []
        for img_lines in images_token :
            max_width = max(line.shape[0] for line in img_lines)
            padded_lines = [np.pad(line, ((0, max_width - line.shape[0]), (0, 0)), 'constant') for line in img_lines]
            img_list.append(np.stack(padded_lines))
        return img_list
    

### basic test
if __name__ == '__main__' :
    data_reduction = {"image":2, "audio":4}
    tokenizer = ActionPixelBytesTokenizer(data_slicing=data_reduction)
    from datasets import load_dataset
    pxby_dataset = load_dataset("ffurfaro/PixelBytes-PokemonAll")
    bulbi = img = pxby_dataset['train']['image'][0]
    text = pxby_dataset['train']['text'][0]
    cry = pxby_dataset['train']['audio'][0]
    bulbi_ids = tokenizer(text=text, image=img, audio=cry)
    print(bulbi_ids)




