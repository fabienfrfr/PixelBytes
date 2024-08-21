#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr

TESTS FILE
"""

from mamba_bys.pokemon import * 

### basic test
if __name__ == '__main__' :
    # get miniatures
    miniatures = get_pkmns_miniatures()
    # construct dataset
    dataset = create_pkmn_dataset(miniatures)
    # push to hub
    push_dataset(dataset, 'ffurfaro/PixelBytes-Pokemon')
    # show
    img = dataset['image'][0] # Image.open(io.BytesIO(dataset['image'][0]))
    plt.imshow(np.array(img)); plt.show()