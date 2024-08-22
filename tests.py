#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

TESTS FILE
"""

from mamba_bys.mambabys import *
from mamba_bys.train import *
from mamba_bys.dataset import *
from mamba_bys.tokenizer import *

from datasets import load_dataset


### basic test
if __name__ == '__main__' :

    # train using
    hf_dataset = load_dataset("ffurfaro/PixelBytes-Pokemon")
    dataset = PxByDataset(hf_dataset, seq_length=256, stride=64)
    
    pixelbyte = PixelBytesTokenizer()
    vocab_size = len(pixelbyte.vocab)
    embedding_dim = 16
    hidden_dim = 64
    
    model = SimpleSeqModel(vocab_size, embedding_dim, hidden_dim)
    #model = SimpleAttentionModel(vocab_size, embedding_dim, hidden_dim)
    
    # Utilisation
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10,
        save_dir='model'
    )
    
    trainer.train()
    
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
    
    """
    dataset = load_dataset("ffurfaro/PixelBytes-Pokemon")
    pixelbytes = [np.array(pb) for pb in dataset['train']['pixelbyte']]
    tokenizer = PixelBytesTokenizer()

    p = pixelbytes[0]
    tokens = tokenizer.convert_ids_to_tokens(p[:,1,1])
    img = reconstruct_imgs(tokens[:np.random.randint(50,len(tokens)//2)])[0]["image"]
    plt.imshow(img);plt.show()
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
