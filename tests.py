#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

TESTS FILE
"""

from mamba_bys.dataset import * 
from datasets import load_dataset

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
    print(np.array(dataset['train']['pixelbyte'][0]).shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
