#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import torch, unicodedata
from torch import nn
from transformers import PreTrainedTokenizer
from skimage.color import rgb2lab
from typing import List, Dict, Union, Tuple
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats

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
DEFAULT_ACTION_STATE = np.linspace(-1, 1, 38).tolist()

##### Tokenizer
class ActionPixelBytesTokenizer(PreTrainedTokenizer):
    def __init__(self, data_slicing=1, **kwargs):
        ## Bytes (ASCII - UTF8) + Pixel (RGB NES Palette) + Action-space (Control & Audio)
        ActionPixelbytes = DEFAULT_BYTES + DEFAULT_PALETTE + DEFAULT_ACTION_STATE
        self.vocab = {ActionPixelbytes[i]: i for i in range(len(ActionPixelbytes))}
        super().__init__(**kwargs)
        self.slicing = data_slicing
        self.bytes_size = len(DEFAULT_BYTES)
        self.palet_size = len(DEFAULT_PALETTE)
        self.LabPalette = torch.tensor(rgb2lab(np.array(DEFAULT_PALETTE)[None, :, :] / 255.0)[0])
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
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        
        text_tokens, image_frames, current_frame, audio_tokens = [], [], [], []
        for token in tokens:
            if isinstance(token, bytes):
                if token == b'\n':
                    if current_frame:
                        image_frames.append(current_frame)
                        current_frame = []
                elif token in DEFAULT_BYTES:
                    text_tokens.append(token)
            elif isinstance(token, tuple):
                current_frame.append(token)
            elif isinstance(token, (int, float)):
                audio_tokens.append(token)
        if current_frame:
            image_frames.append(current_frame)
        return {'text': text_tokens,
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
        frames = (frames + self.bytes_size)[::self.slicing, ::self.slicing, ::self.slicing]
        added_column = torch.ones((frames.shape[0], frames.shape[1], 1), dtype=torch.long)
        added_column[:, -1, :] = 2
        return torch.cat((frames, added_column), dim=2)

    def process_action_state(self, audio):
        audio = torch.tensor(audio)
        if audio.dim() < 2:
            audio = audio.unsqueeze(0)
        normalized_state = torch.clamp(torch.tensor(stats.zscore(audio.numpy(), axis=1)), -1, 1).flatten()
        action_state_tensor = torch.tensor(DEFAULT_ACTION_STATE).reshape(-1, 1)
        indices = torch.tensor(cKDTree(action_state_tensor.numpy()).query(normalized_state.reshape(-1, 1).numpy())[1]).reshape(audio.shape)
        indices = (indices + self.bytes_size + self.palet_size)[:, ::self.slicing]
        separators = torch.tensor([[1], [2]])[:indices.shape[0]]
        return torch.cat([indices, separators], dim=1).T.unsqueeze(1)

    def create_sequence_data(self, context_array):
        n_frames, height, width = context_array.shape
        padded = nn.functional.pad(context_array, (1, 1, 1, 1, 1, 0), mode='constant', value=0)
        slices = [
            padded[:-1, :-2, 1:-1], # (t-1, i-1, j) : up value in t-1
            padded[:-1, 1:-1, 1:-1],# (t-1, i, j) : value in t-1
            padded[:-1, 2:, 1:-1],  # (t-1, i+1, j) : down value in t-1
            padded[1:, :-2, :-2],   # (t, i-1, j-1) : up-left value in t
            padded[1:, 1:-1, :-2],  # (t, i, j-1) : up value in t
            padded[1:, :-2, 1:-1]]   # (t, i-1, j) : left value in t
        context = torch.stack(slices, dim=-1)
        return context.reshape(-1, 6), context_array.reshape(-1, 1)

### basic test
if __name__ == '__main__' :
    tokenizer = ActionPixelBytesTokenizer()
    from datasets import load_dataset
    pxby_dataset = load_dataset("ffurfaro/PixelBytes-PokemonAll")
    bulbi = img = pxby_dataset['train']['image'][0]
    text = pxby_dataset['train']['text'][0]
    cry = pxby_dataset['train']['audio'][0]
    bulbi_ids = tokenizer(text=text, image=img, audio=cry)
    print(bulbi_ids)




