.. -*- mode: rst -*-

==========
Mamba-Bys
==========

Mamba-Bys is a Python project aimed at testing byte bidirectional Mamba models (particularly those with shared SSMs and token-free) on tasks beyond multimodal processing.

Hypothesis
----------

Bidirectional byte models allow for greater generalization compared to unidirectional models with using one input for all modality.

Objectives
----------

The main objectives of this project are:

1. Construct captionning database.
2. Implement image captioning using Mamba architecture.

Pokemon database
^^^^^^^

https://huggingface.co/datasets/ffurfaro/PixelBytes-Pokemon

Image Captioning
^^^^^^^^^^^^^^^^

In the case of image captioning, we will assign "image tokens" and "text tokens" to test scenarios where input modalities are mixed. (This approach may be somewhat complex.)


Getting Started
---------------

(Add installation instructions and basic usage examples here)

Training in a Notebook
----------------------

To facilitate experimentation and visualization, we recommend using Jupyter notebooks for training. Here are some suggested steps for setting up a training notebook:

1. **Environment Setup**:
   
   - Import necessary libraries (PyTorch, etc.)
   - Set up CUDA if using GPU acceleration

2. **Data Preparation**:
   
   - For Othello: Create a dataset of game states and moves
   - For Image Captioning: Prepare image-text pairs dataset

3. **Model Definition**:
   
   - Define the Mamba model architecture

4. **Training Loop**:
   
   - Define loss function and optimizer
   - Implement training and validation loops
   - Add checkpointing for model saving

5. **Visualization**:
   
   - Plot training and validation losses
   - For Othello: Visualize game board states
   - For Image Captioning: Display sample captions for test images

6. **Hyperparameter Tuning**:
   
   - Use tools for hyperparameter optimization

7. **Evaluation**:
   
   - Implement metrics specific to each task (e.g., win rate for Othello, BLEU score for captioning)

Example Notebook Structure::

    # 1. Setup
    import torch
    from mamba_bis import Mamba
    
    # 2. Data Preparation
    # (Task-specific data loading code)
    
    # 3. Model Definition
    model = Mamba(...)
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        # Training code
        # Validation code
    
    # 5. Visualization
    # (Plotting code)
    
    # 6. Hyperparameter Tuning
    # (Optuna setup if needed)
    
    # 7. Evaluation
    # (Task-specific evaluation code)

Contributing
------------

(Add guidelines for contributing to the project)

License
-------

MIT Licence

Contact
-------

(Provide contact information or links to project maintainers)
