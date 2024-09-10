#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabien

in dev (pretrained part)
"""

from huggingface_hub import login, hf_hub_download

from transformers import PreTrainedTokenizer
import numpy as np, os
from typing import List, Dict, Union, Tuple

##### Tokenizer
class PixelBytesTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None):
        if vocab == None :
            Pixelbytes_tokens =  [
                ## Bytes (ASCII - UTF8)
                b'\x00', b'\t', b'\n', b' ', b'"', b"'", b'(', b')', b'*', b',', b'-', b'+', 
                b'.', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'\xc2', 
                b'\xa0', b':', b'[', b']', b';', b'/', b'%', b'!', b'a', b'b', b'c', b'd', b'e', 
                b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', 
                b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
                ## Pixel (RGB NES Palette)
                (0x00, 0x00, 0x00), (0xfc, 0xfc, 0xfc), (0xf8, 0xf8, 0xf8), (0xbc, 0xbc, 0xbc),
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
            vocab = {Pixelbytes_tokens[i] : i for i in range(len(Pixelbytes_tokens))}
        self.vocab = vocab
        super().__init__()
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return {k: v for k, v in self.vocab.items()}

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: Union[bytes, tuple]) -> int:
        return self.vocab.get(token, self.vocab.get(b'[UNK]', 0))

    def _convert_id_to_token(self, index: int) -> Union[bytes, tuple]:
        return self.ids_to_tokens.get(index, b'[UNK]')

    def convert_tokens_to_ids(self, tokens: List[Union[bytes, tuple]]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[Union[bytes, tuple]]:
        return [self._convert_id_to_token(i) for i in ids]
    
    # change here : convert byte in b"\t" in line (use ast.literal_eval(f"'{s}'"))
    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> Tuple[str]:
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + 'vocab.txt')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token, index in self.vocab.items():
                if isinstance(token, bytes):
                    token = token.decode('utf-8', errors='replace')
                elif isinstance(token, tuple):
                    token = ','.join(map(str, token))
                f.write(f"{token}\n")
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        try:
            vocab_file = hf_hub_download(repo_id=pretrained_model_name_or_path,
                                         filename="vocab.txt",
                                         subfolder=kwargs.get("subfolder", None))
        except Exception as e:
            return cls()
        vocab = cls._load_vocab(vocab_file)
        print(f"Taille du vocabulaire chargé: {len(vocab)}")
        return cls(vocab=vocab, **kwargs)
    
    # readapting with save vocab --> no save with binary
    @staticmethod
    def _load_vocab(vocab_file):
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    if ',' in line: # error ici si c'est une virgule
                        token = tuple(map(int, line.split(',')))
                    elif len(line) == 1 or (len(line) == 2 and line.startswith('\\')):
                        token = line.encode('utf-8')
                    else:
                        token = line
                    vocab[token] = i
        return vocab

def push_tokenizer_to_hub(tokenizer, repo_name="ffurfaro/PixelBytes-Pokemon"):
    token = input("Input Hugging Face Token: ")
    # Connect and push to Hub
    login(token)
    tokenizer.push_to_hub(repo_name, use_auth_token=token)

