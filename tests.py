#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
TESTS FILE
"""

from pixelbytes.model import *
from pixelbytes.train import *
from pixelbytes.dataset import *
from pixelbytes.tokenizer import *

from datasets import load_dataset


### basic test
if __name__ == '__main__' :

    # train using
    hf_dataset = load_dataset("ffurfaro/PixelBytes-Pokemon")
    ds = hf_dataset["train"].train_test_split(test_size=0.1)
    
    train_dataset = PxByDataset(ds["train"]["pixelbyte"], seq_length=256, stride=128)
    test_dataset = PxByDataset(ds["test"]["pixelbyte"], seq_length=256, stride=128)
    
    pixelbyte = PixelBytesTokenizer()
    vocab_size = len(pixelbyte.vocab); print(vocab_size)
    embedding_dim = 81
    hidden_dim = 64
    n_layer = 1
    
    config = ModelConfig(dim=embedding_dim, d_state=hidden_dim, depth=n_layer, vocab_size=vocab_size)
    #model = SimpleRNNModel(config)
    model = SimpleTransformerModel(config)
    """
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        outputs = model(inputs)
        print(inputs[:,-1,1,1], outputs.max(1)[1])
        break
    
    """
    # Utilisation
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=12,
        save_dir='models',
        model_name="SimpleSeqModel",
        dataset_name="PixelBytes-Pokemon",
        eval_every=1 # Évaluer tous les 5 epochs
    )
    
    trainer.train_and_evaluate()
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
