.. -*- mode: rst -*-


|GitHub|_ |PyPi|_ |DOI|_


.. |GitHub| image:: https://img.shields.io/github/v/release/fabienfrfr/functionalfilet
.. _GitHub: https://github.com/fabienfrfr/functionalfilet

.. |PyPi| image:: https://img.shields.io/pypi/v/functionalfilet
.. _PyPi: https://pypi.org/project/functionalfilet


.. |DOI| image:: https://img.shields.io/badge/arXiv-ANNFE-%3CCOLOR%3E.svg
.. _DOI: https://arxiv.org/abs/2205.10118


.. |PythonMinVersion| replace:: 3.5
.. |PyTorchMinVersion| replace:: 1.0.0


.. image:: https://raw.githubusercontent.com/fabienfrfr/Mamba-MMix/main/branding/assets/MambaMMix.png
  :target: https://pypi.org/project/functionalfilet/


**Mamba-MMix** is a Python project using mambapy for exploration of multimodal mixed training of Mamba backbone... 

This project try to reintroduce multibatch algorithms like Gato, algorithms which have not been successful with transformers and with the advent of LLMs, but which have potential with Mamba due to its sequential aspect. All while remaining within a minimalist prediction logic such as token generation.

Installation
------------

Dependencies
~~~~~~~~~~~~

Functional-Filet requires:

- Python (>= |PythonMinVersion|)
- NumPy
- Pandas
- PyTorch (>= |PyTorchMinVersion|)
- Torchvision
- Matplotlib
- Networkx

Optionally, you need:

- Scikit-learn
- Seaborn
- Gym

=======

Functional-Filet is stable only from version 0.5.2, any previous version corresponds to the development phase.

However, there are several possible optimizations, in particular on the restructuring of *Torch* tensors in Python which could be done in C++. For this, it is possible that there will be several code modifications in the future.


The documentation includes more detailed `installation and examples instructions <https://github.com/fabienfrfr/functionalfilet/blob/main/doc/notebook.ipynb>`_.


User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and pytorch,
the easiest way to install scikit-learn is using ``pip``::

    python3 -m pip install functionalfilet


Development
-----------

We welcome new contributors of all experience levels.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/fabienfrfr/functionalfilet
- Download releases: https://pypi.org/project/functionalfilet

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/fabienfrfr/functionalfilet.git


Utilization
-----------

Once installed, if we consider two variables *feature X* and *label y* already executed upstream of the code, here is a simple example of use in the case of a classification problem::

	# package
	from functionalfilet import model as ff 
	# init model
	model = ff.FunctionalFilet()
	# train
	model.fit(X,y)
	# test
	y_pred = model.predict(X, index=seeder_idx)


Existing code
~~~~~~~~~~~~~

There is in the *example* directory of the git, several code to play with the learning parameters in simple cases. A brief summary is described at the top of each file::

	python3 -m IPython
	# universal approximation theorem
	run example/uat_regression.py
	# classification with overlapping and unbalance
	run example/blob_classification.py
	# reinforcment leaning with time dependancy
	run example/gym_RL-CartPole-v0.py

Citation
~~~~~~~~
If you take inspiration from my machine learning algorithm for a scientific publication, we would appreciate citations::

	@article{furfaro2022artificial,
	title={An Artificial Neural Network Functionalized by Evolution},
	author={Furfaro, Fabien and Bar-Hen, Avner and Berthelot, Geoffroy},
	journal={arXiv preprint arXiv:2205.10118},
	year={2022}
	}

**Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)**
