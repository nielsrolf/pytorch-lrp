import torch.nn.functional as F
from torch import nn
import numpy as np
import copy
from trainable_net import TrainableNet

na = None


class ExplainableLayer():
    """
    Base class for all layers that support deeptaylor
    """
    @classmethod
    def from_module(cls, src):
        layer = copy.deepcopy(src)
        layer.__class__ = cls
        return layer


class ReluLayer(ExplainableLayer, nn.Module):
    """
    Base class for layers that use Relus as nonlinearity.
    """
    def forward(self, x):
        self.x = x
        self.a = super().forward(x) # Super is called from a child class and via dependency injection this will be eg nn.Linear's forward layer 
        self.z = F.relu(self.a)
        return self.z
        
    
class Linear(ExplainableLayer, nn.Linear):
    """
    For linear layers without Relu activation, usually the last
    """
    def forward(self, x):
        D = np.prod(x.shape[1:])
        self.x = x
        x = x.view(-1, D)
        self.a = super().forward(x) # Super is called from a child class and via dependency injection this will be eg nn.Linear's forward layer 
        self.z = self.a
        return self.z
    
    def zplus(self, R, eps=1e-9):
        w_plus = F.relu(self.weight) # (D, d)
        z_plus = F.linear(self.x, w_plus, None)
        R_norm = R/(z_plus+eps) # (N, d)
        return F.linear(R_norm, w_plus.transpose(0, 1))*self.x
    
    def zb(self, R, l, h, eps=1e-9):
        w_plus  = F.relu(self.weight)
        w_minus = -F.relu(-self.weight)
        norm = self.z - l*F.linear(self.x, w_plus) - h*F.linear(self.x, w_minus)
        R_norm = R/norm
        return F.linear(R_norm, self.weight.transpose(0, 1))              - l*F.linear(R_norm, self.w_plus.transpose(0, 1))             - h*F.linear(R_norm, self.w_minus.transpose(0, 1))


class LinearRelu(ReluLayer, Linear):
    """
    Combines the forwarding behaviour of ReluLayer with the
    relevance propagation rule of Linear
    """
    def forward(self, x):
        D = np.prod(x.shape[1:])
        return super().forward(x.view(-1, D))
    
    
class ConvRelu(ReluLayer, nn.Conv2d):
    """
    Implements deep taylor for conv2d layers
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_args = {
            'padding': self.padding,
            'stride': self.stride
        }
        
    def zplus(self, R, eps=1e-9):
        w_plus = F.relu(self.weight) # (D, d)
        z_plus = F.conv2d(self.x, w_plus, **self.conv_args)
        R_norm = R/(z_plus+eps) # (N, d)
        R_out_norm = F.conv_transpose2d(R_norm, w_plus, **self.conv_args)
        return R_out_norm*self.x
    
    def zb(self, R, l, h, eps=1e-9):
        w_plus  = F.relu(self.weight)
        w_minus = -F.relu(-self.weight)
        l = self.x*0 + l
        h = self.x*0 + h
        norm = F.conv2d(self.x, self.weight, **self.conv_args)             - F.conv2d(l, w_plus, **self.conv_args)             - F.conv2d(h, w_minus, **self.conv_args)
        R_norm = R/norm
        return self.x*F.conv_transpose2d(R_norm, self.weight, **self.conv_args)              - l*F.conv_transpose2d(R_norm, w_plus, **self.conv_args)             - h*F.conv_transpose2d(R_norm, w_minus, **self.conv_args)
    
    @classmethod
    def from_module(cls, src):
        layer = copy.deepcopy(src)
        layer.__class__ = cls
        layer.conv_args = {
            'padding': layer.padding,
            'stride': layer.stride
        }
        return layer
    

class MaxPool(ReluLayer, nn.MaxPool2d):
    """
    Maxpool2d layer with the interface of the explainable ReluLayer
    """
    def __init__(self, *args, **kwargs):
        kwargs['return_indices'] = True
        super().__init__(*args, **kwargs)
        self.unpool = nn.MaxUnpool2d(kernel_size=self.kernel_size, stride=self.stride)
        
    def forward(self, x):
        self.x = x
        self.a, self.indices = nn.MaxPool2d.forward(self, x) 
        self.z = F.relu(self.a)
        return self.z
    
    def zplus(self, R, eps=1e-9):
        # distribute relevance flat to all input fields
        return self.unpool(R, self.indices, output_size=self.x.size())
    
    @staticmethod
    def from_module(src):
        return MaxPool(kernel_size=src.kernel_size,
                      stride=src.stride,
                      padding=src.padding,
                      dilation=src.dilation,
                      return_indices=True,
                      ceil_mode=src.ceil_mode
                      )
    
    
class ExplainableModel(TrainableNet):
    """
    Base class for networks that implement deep taylor decomposition.
    Network architecture is not specified here
    
    Subclasses can do some of the following:
        - Add an architecture to `__init__`
        - Overwride `deeptaylor,` if the architecture and input domains require a
        different set of rules
        - Overwride `from_model` if the architecture requires it.
    """
    def __init__(self, layers, min_x=None, max_x=None):
        """
        Layers: Instances of subclasses of ReluLayer
        min_x, max_x: box constraints of the input domain, zB rule will be used there
            if not set, ww rule will be used in the first layer
        """
        super().__init__()
        self.layers = layers
        self.min_x, self.max_x = min_x, max_x
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def layer_shapes(self, x):
        print(x.shape)
        for layer in self.layers:
            x = layer.forward(x)
            print(f"({layer.__class__.__name__}) ->", x.shape)
        return x
    
    def deeptaylor(self, x, pattern):
        """
        Deep Taylor Decomposition, where relevance is the outputlayer is predictions*pattern
        Pattern: eg one_hot(labels) if the heatmap for the correct class should be used
        """
        def debug_info(layer, R):
            print(f"({layer.__class__.__name__}) ->", R.shape)
            print(f"({layer.__class__.__name__})     Conservation quotient: {R.sum()/R_total}")
            if R.min() < 0:
                print(f"({layer.__class__.__name__})     encountered negative values")
            print("")
            
        out = self.forward(x)
        R = F.relu(out*pattern)
        R_total = R.sum()
        debug_info(self, R)
            
        for layer in self.layers[1:][::-1]:
            R = R.view(layer.z.shape)
            R = layer.zplus(R)
            debug_info(layer, R)
            
        # Select the input layer rule
        if self.min_x is None and self.max_x is None:
            R = self.layers[0].ww(R)
        elif self.min_x == 0 and self.max_x is None:
            R = self.layers[0].zplus(R)
        else:
            R = self.layers[0].zb(R, self.min_x, self.max_x)
        debug_info(self.layers[0], R)
        return R
    
    @classmethod
    def from_model(cls, model, layer_names, *args, **kwargs):
        """
        model: nn.Module
        layer_names: list[string] contains the property names of the mappings that define the network
        eg: if model.forward does this:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        layer_names should look like this:
        ['conv1', 'pool', 'conv2', 'pool', 'fc1', 'fc2', 'fc3']
        
        Relus don't have to be specified:
            assumes that relus are used everywhere but in the last layer
            assumes the last layer is linear
        Reshaping does not have to be specified
        
        Relies on the fact that each ReluLayer is also subclass of the class of model
        """
        def casted_layer(src, last_layer=False):
            # finds the Explainable equivalent of src.__class__
            if last_layer:
                return Linear.from_module(src)
            for layer_type in ReluLayer.__subclasses__():
                if issubclass(layer_type, src.__class__):
                    return layer_type.from_module(src)
        
        # cast all layers
        # all but the last are casted to relu layers
        layers = [casted_layer(model._modules[l]) for l in layer_names[:-1]] +                  [casted_layer(model._modules[layer_names[-1]], last_layer=True)]
        
        # now create a ExplainBase object which we call self, and pretend to be a constructor
        self = cls(layers, *args, **kwargs)
        # add the layers to the properties of self, in hope that 
        # pytorch transforms it to a self._modules element
        for l, layer in zip(layer_names, layers):
            self.__dict__[l] = layer
            
        return self



class ExplainNet(ExplainableModel):
 """
 Rebuilt of cifar10utils.Net with deep taylor support
 """
 def __init__(self):
     layers = [
         ConvRelu(3, 6, 5),
         MaxPool(2, 2),
         ConvRelu(6, 16, 5),
         MaxPool(2, 2),
         LinearRelu(16 * 5 * 5, 120),
         LinearRelu(120, 84),
         LinearRelu(84, 10)
     ]
     super().__init__(layers, -1, 1)
     
     # Because pytorch is weird, the submodels cannot be stored in a list
     self.l1 = layers[0]
     self.l2 = layers[1]
     self.l3 = layers[2]
     self.l4 = layers[3]
     self.l5 = layers[4]
     self.l6 = layers[5]
