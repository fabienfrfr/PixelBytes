.. -*- mode: rst -*-

==========
PixelBytes
==========

PixelByte is an innovative Python project designed to simultaneously generate text and images pixel by pixel in the form of sequences. The goal is to explore a unified embedding that allows for coherent multimodal generation.


Installation
------------

To get started with this project, you need to have Python 3.8 or higher installed. It is recommended to use a virtual environment.


Install dependencies using PyPI:

.. code-block:: bash

   pip install git+https://github.com/fabienfrfr/PixelBytes.git@main



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

1. Implement a multimodal sequence generation model (PixelByte).
2. Test the model on tasks beyond multimodal processing, including image and text generation.
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


Available Tasks
~~~~~~~~~~~~~~~

- **Build the dataset**:

.. code-block:: bash

   python -m your_package.main build --path /path/to/dataset --palette /path/to/palette.py

- **Train the model**:

.. code-block:: bash

   python -m your_package.main train --model resnet50 --learning-rate 0.0001 --batch-size 64 --epochs 20

- **Evaluate the model**:

.. code-block:: bash

   python -m your_package.main evaluate --metrics accuracy precision recall

- **Generate output**:

.. code-block:: bash

   python -m your_package.main generate --format png

If no task is specified, the script will display help information. You can see docs for manual using (in French) in the `docs folder <docs/>`_.

Tasks
-----

Build Dataset
~~~~~~~~~~~~~

This task builds a dataset from the specified path and optionally uses a custom palette.

Train Model
~~~~~~~~~~~

This task trains the specified model with the provided hyperparameters, including learning rate, batch size, and number of epochs.

Evaluate Model
~~~~~~~~~~~~~~

This task evaluates the model using the specified metrics.

Generate Output
~~~~~~~~~~~~~~~

This task generates and displays results in the specified format (e.g., SVG, PNG, JPG).


Contributing
------------

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

1. Fork the repository.
2. Create your feature branch (``git checkout -b feature/AmazingFeature``).
3. Commit your changes (``git commit -m 'Add some AmazingFeature'``).
4. Push to the branch (``git push origin feature/AmazingFeature``).
5. Open a pull request.

License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

Contact
-------

fabien.furfaro_at_gmail.com
