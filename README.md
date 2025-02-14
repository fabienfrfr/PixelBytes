.. -*- mode: rst -*-

PixelBytes+
===========

PixelBytes+ is an Python project that generates and processes multimodal sequences, including pixels/video, audio, action-states, and text in a unified representation.

Installation
------------

Requires Python 3.8+. Install via PyPI:

.. code-block:: bash

   pip install git+https://github.com/fabienfrfr/PixelBytes.git@main


Overview
--------

PixelBytes+ builds on theoretical foundations including Image Transformers, PixelRNN/PixelCNN, Bi-Mamba+, and MambaByte to create a unified representation for coherent multimodal generation and processing. It handles:

- Pixel/video sequences
- Audio data
- Action-state control
- Text

The model seamlessly manages transitions between modalities and maintains dimensional consistency.

Usage
-----

Basic commands :

.. code-block:: python

    tokenizer = ActionPixelBytesTokenizer(data_slicing=DATA_REDUCTION)
    config = ModelConfig(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, 
                          num_layers=NUM_LAYERS, pxby_dim=PXBY_DIM, auto_regressive=AR, model_type=MODEL_TYPE)
    model = aPxBySequenceModel(config).to(DEVICE)
    dataset = TokenPxByDataset(ds, tokenizer, SEQ_LENGTH, STRIDE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    model.train_model(train_dataloader, val_dataloader, optimizer, criterion, DEVICE, scaler, EPOCHS, ACCUMULATION_STEPS)


For detailed documentation, see the `docs folder <docs/>`_.

Dataset
-------

Use the PixelBytes-Pokemon dataset from Hugging Face: `ffurfaro/PixelBytes-Pokemon <https://huggingface.co/datasets/ffurfaro/PixelBytes-Pokemon>`_

Cloud Deployment
----------------

Build and push Docker image:

.. code-block:: bash

   docker build -t $USER/img_name .
   docker push $USER/img_name

   docker-compose up --build

Deploy to your preferred cloud provider (OVH, Azure, AWS, Google Cloud).

Contributing
------------

Contributions welcome. Fork, create a feature branch, and submit a pull request.

License
-------

MIT License

Contact
-------

fabien.furfaro_at_gmail.com

Citation
--------

.. code-block:: bibtex

   @article{furfaro:hal-04683349,
     TITLE = {{PixelBytes: Catching Unified Representation for Multimodal Generation}},
     AUTHOR = {Furfaro, Fabien},
     URL = {https://hal.science/hal-04683349},
     NOTE = {working paper or preprint},
     YEAR = {2024},
     KEYWORDS = {Embedding ; Multimodal representation learning ; Sequence generation},
     HAL_ID = {hal-04683349},
   }

   @misc{furfaro2024pixelbytes_project,
        author = {Furfaro, Fabien},
        title = {PixelBytes: A Unified Multimodal Representation Learning Project},
        year = {2024},
        howpublished = {
            GitHub: \url{https://github.com/fabienfrfr/PixelBytes},
            Models: \url{https://huggingface.co/ffurfaro/PixelBytes-Pokemon} and \url{https://huggingface.co/ffurfaro/aPixelBytes-Pokemon},
            Datasets: \url{https://huggingface.co/datasets/ffurfaro/PixelBytes-Pokemon} and \url{https://huggingface.co/datasets/ffurfaro/PixelBytes-PokemonAll}
        },
        note = {GitHub repository, Hugging Face Model Hub, and Datasets Hub}
        }


