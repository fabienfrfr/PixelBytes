.. -*- mode: rst -*-

==========
Mamba-Bis
==========

Mamba-Bis is a Python project aimed at testing bidirectional Mamba models (particularly those with shared SSMs) on tasks beyond image processing.

Hypothesis
----------

Bidirectional models allow for greater generalization compared to unidirectional models.

Objectives
----------

The main objectives of this project are:

1. Train Mamba blocks to solve Othello (or potentially just Othelo).
2. Implement image captioning using Mamba architecture.

Othello
^^^^^^^

For the Othello implementation, we will draw inspiration from the project:

https://github.com/alxndrTL/othello_mamba

Image Captioning
^^^^^^^^^^^^^^^^

In the case of image captioning, we will assign "image tokens" and "text tokens" to test scenarios where input modalities are mixed. (This approach may be somewhat complex.)

Implementation
--------------

To simplify the implementation of State Space Models (SSMs), we will utilize the Zeta library:

https://github.com/kyegomez/zeta

Getting Started
---------------

(Add installation instructions and basic usage examples here)

Training in a Notebook
----------------------

To facilitate experimentation and visualization, we recommend using Jupyter notebooks for training. Here are some suggested steps for setting up a training notebook:

1. **Environment Setup**:
   
   - Import necessary libraries (PyTorch, Zeta, etc.)
   - Set up CUDA if using GPU acceleration

2. **Data Preparation**:
   
   - For Othello: Create a dataset of game states and moves
   - For Image Captioning: Prepare image-text pairs dataset

3. **Model Definition**:
   
   - Define the Mamba model architecture using Zeta

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
    from zeta import Mamba
    
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

(Specify the license under which the project is released)

Contact
-------

(Provide contact information or links to project maintainers)
