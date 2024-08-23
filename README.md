.. -*- mode: rst -*-

==========
PixelBytes
==========

PixelByte is an innovative Python project designed to simultaneously generate text and images pixel by pixel in the form of sequences. The goal is to explore a unified embedding that allows for coherent multimodal generation.

Context and Proposed Architecture
----------------------------------

Theoretical Foundations
----------------

- **Image Transformer**: [Pixel-by-pixel image generation](https://arxiv.org/abs/1802.05751)
- **Bi-Mamba+**: [Bidirectional model for time series forecasting](https://arxiv.org/abs/2404.15772)
- **MambaByte**: [Token-free selective state space model](https://arxiv.org/abs/2401.13660)

Key Concept
----------------

The PixelByte model generates mixed sequences of text and images. It aims to:
- Handle transitions between text and image with line breaks (ASCII 0A).
- Maintain consistency in the dimensions of generated images.
- Master the "copy" task to reproduce complex patterns.

This project utilizes the power of two T4 GPUs from Kaggle to experiment with advanced architectures and large datasets, tackling the challenges of unified multimodal generation.

Objectives
---------

The main objectives of this project are:

1. Implement a multimodal sequence generation model (PixelByte) using the Mamba architecture.
2. Test the model on tasks beyond traditional multimodal processing, including image generation and image captioning.
3. Explore the model's ability to seamlessly manage transitions between text and image modalities.

Project Resources
-----------------

Dataset
----------------

For this project, we will use the **PixelBytes-Pokemon** dataset, specifically designed for this multimodal generation task. This dataset, created by the author of this notebook, is available on Hugging Face: [PixelBytes-Pokemon](https://huggingface.co/datasets/ffurfaro/PixelBytes-Pokemon). It contains sequences of text and Pokémon images, encoded to enable training of our PixelByte model on multimodal data.

Implementation
----------------

The model implementation and training scripts are available in this GitHub repository **PixelBytes**: [PixelBytes](https://github.com/fabienfrfr/PixelBytes). This repository contains the source code necessary to reproduce the experiments, as well as detailed instructions on configuring and using the PixelByte model.


Citation
========

If you use this package in your research or project, please cite it as follows:

.. code-block:: bibtex

    @misc{pixelbytes,
        author = {Fabien Furfaro},
        title = {PixelBytes: Catching Insights in Unified Multimodal Sequences},
        year = {2024},
        publisher = {GitHub},
        journal = {GitHub repository},
        url = {https://github.com/fabienfrfr/PixelBytes}
    }


Image Captioning
----------------

In the context of image generation, we assign "image tokens" and "text tokens" to test scenarios where input modalities are mixed, exploring a more complex approach to multimodal processing.

Getting Started
---------------

[Add your getting started instructions here]


Contributing
------------

(Add guidelines for contributing to the project)

License
-------

MIT Licence

Contact
-------

(Provide contact information or links to project maintainers)
