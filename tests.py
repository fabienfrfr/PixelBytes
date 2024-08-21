#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

TESTS FILE
"""

from mamba_bys.dataset import *  ## creer un 
from datasets import load_dataset

from torch.utils.data import Dataset
import torch


### https://www.perplexity.ai/search/j-ai-un-dataset-huggingface-av-MQCEI6QySEuHt7RsC2cr4g 
class SequenceDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset["train"]["pixelbyte"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        return torch.tensor(sequence)

def collate_fn(batch):
    # Trier les séquences par longueur décroissante
    batch.sort(key=lambda x: x.shape[0], reverse=True)
    
    # Padding
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    
    # Créer un masque pour indiquer les éléments de padding
    mask = (padded_batch != 0).float()
    
    return padded_batch, mask

from torch.utils.data import DataLoader

# Créer le dataset personnalisé
dataset = SequenceDataset(hf_dataset)

# Créer le DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

for batch, mask in dataloader:
    # batch est un tensor de forme [batch_size, max_seq_length = 512, 3, 3]
    # mask est un tensor de forme [batch_size, max_seq_length = 512, 3, 3]
    
    # Votre logique de traitement ici
    ...


### basic test
if __name__ == '__main__' :

    """
    ## Construct Pokemon datasets
    # get miniatures
    miniatures = get_pkmns_miniatures()
    # construct dataset
    dataset = create_pkmn_dataset(miniatures)
    # push to hub
    #push_dataset(dataset, 'ffurfaro/PixelBytes-Pokemon')
    # show
    img = dataset['image'][0] # Image.open(io.BytesIO(dataset['image'][0]))
    plt.imshow(np.array(img)); plt.show()
    """
    """
    dataset = load_dataset("ffurfaro/PixelBytes-Pokemon")
    dataset = dataset.remove_columns('name')
    dataset = add_pixelbyte_columns(dataset)
    # push to hub
    push_dataset(dataset, 'ffurfaro/PixelBytes-Pokemon')
    """
    
    dataset = load_dataset("ffurfaro/PixelBytes-Pokemon")
    
    pixelbytes = [np.array(pb) for pb in dataset['train']['pixelbyte']]
    tokenizer = PixelBytesTokenizer()

    p = pixelbytes[0]
    tokens = tokenizer.convert_ids_to_tokens(p[:,1,1])
    img = reconstruct_imgs(tokens[:np.random.randint(50,len(tokens)//2)])[0]["image"]
    plt.imshow(img);plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
