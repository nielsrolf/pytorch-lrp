# Pytorch Notebooks

Notebooks I wrote to get familiar with pytorch, and where I try to implement LRP-gamma.

## File contents
- Python files in the order of dependency
  - `trainable_net.py`
    - `TrainableNet`: A class that can be subclassed instead of `torch.nn.Module`, which implements training and evaluating the accuracy
    - Some convenience functions like `one_hot`
    - `input_times_gradient(net, images, target_pattern)`
  - `cifar10_utils.py` <- `trainable_net.py`
    -  code of LeNet taken from [the pytorch cifar10 tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-download-beginner-blitz-cifar10-tutorial-py).
    -  Some convenience methods like `one_hot`
    -  A function to plot heatmaps
  - `layerized_net.py` <- `trainable_net.py`
    - A `Layer` and a `LayerizedNet` class that convert existing models to layerized versions. The layer types have to be implemented for the specific use case.
  - `lrp.py` <- `layerized_net.py` Layers and a network class that support deep taylor decomposition and LRP gamma.
- Notebooks
-  `save_load_example.ipynb` is a minimal demo of how to use the net. It trains, saves and loads the net.
-  `lrp-gamma.ipynb` <- `layerized_net.py`, `cifar10_utils.py` Implementation and a few tests with [deep taylor decomposition](arxiv.org/pdf/1512.02479.pdf) and lrp-gamma on cifar10. Here is where `lrp.py` comes from
-  `mnist_deeptaylor.ipynb` <- `lrp.py` MNIST classifier and DTD usage, example how to use `lrp.py`
-  `cifar10_adversarial_examples.ipynb` generates adversarial examples. They have been introduced [here](arxiv.org/abs/1312.6199)
-  `gradient_times_input.ipynb` <- `trainable_net.py`, `cifar10_utils.py` Example with cifar10