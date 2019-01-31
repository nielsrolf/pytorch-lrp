# Pytorch Notebooks

Notebooks I wrote to get familiar with pytorch, and where I try to implement LRP-gamma.

## File contents
- `cifar10_utils.py`
  -  code of LeNet taken from [https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-download-beginner-blitz-cifar10-tutorial-py](the pytorch cifar10 tutorial).
  -  Some convenience methods like `one_hot`
  -  A function to plot heatmaps
-  `save_load_example.ipynb` is a minimal demo of how to use the net. It trains, saves and loads the net.
-  `cifar10_adversarial_examples.ipynb` generates adversarial examples. They have been introduced [arxiv.org/abs/1312.6199](here)
-  `lrp-gamma.ipynb` is the notebook I'm working on know, where I first implement [arxiv.org/pdf/1512.02479.pdf](deep taylor decomposition) and then lrp-gamma.