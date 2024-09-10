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

Basic commands (see old version) :

.. code-block:: bash

   python -m your_package.main build --path /path/to/dataset
   python -m your_package.main train --model rnn --learning-rate 0.0001
   python -m your_package.main evaluate --metrics accuracy
   python -m your_package.main generate --format png

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

   @misc{pixelbytes,
   author = {Fabien Furfaro},
   title = {PixelBytes+: Unified Multimodal Sequence Processing},
   year = {2024},
   publisher = {GitHub},
   journal = {GitHub repository},
   url = {https://github.com/fabienfrfr/PixelBytes}
   }

