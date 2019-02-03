import torch.nn.functional as F
from torch import nn
import numpy as np
import copy
from layerized_net import Layer, LayerizedNet


na = None


class ExplainableLayer(Layer):
    """
    Base class for all layers that support deeptaylor
    """
    pass


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
        return F.linear(R_norm, self.weight.transpose(0, 1))  \
            - l*F.linear(R_norm, self.w_plus.transpose(0, 1)) \
            - h*F.linear(R_norm, self.w_minus.transpose(0, 1))
    
    def lrp_gamma(self, R, gamma, bias_relevance=False):
        def leaky_relu(a):
            return F.relu(a) - gamma*F.relu(-a)
        w = leaky_relu(self.weight)
        b = leaky_relu(self.bias) if bias_relevance else None
        z = F.linear(self.x, w, b)
        R_norm = R/z
        return F.linear(R_norm, w.transpose(0, 1))*self.x
    


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
        norm = F.conv2d(self.x, self.weight, **self.conv_args) \
            - F.conv2d(l, w_plus, **self.conv_args) \
            - F.conv2d(h, w_minus, **self.conv_args)
        R_norm = R/norm
        return self.x*F.conv_transpose2d(R_norm, self.weight, **self.conv_args)  \
            - l*F.conv_transpose2d(R_norm, w_plus, **self.conv_args) \
            - h*F.conv_transpose2d(R_norm, w_minus, **self.conv_args)
    
    def lrp_gamma(self, R, gamma, bias_relevance=False):
        def leaky_relu(a):
            return F.relu(a) - gamma*F.relu(-a)
        w = leaky_relu(self.weight)
        b = leaky_relu(self.bias) if bias_relevance else None
        z = F.conv2d(self.x, w, b, **self.conv_args)
        R_norm = R/z
        return F.conv_transpose2d(R_norm, w, **self.conv_args)*self.x
    
    def zb_gamma(self, R, l, h, gamma=0.9, bias_relevance=False):
        # I am not sure if the first layer's rule should be
        # the normal zB rule, but this is an interpolation of
        # the zB rule and input*gradient
        
        w_plus  = F.relu(self.weight)
        w_minus = -F.relu(-self.weight)
        l = self.x*0 + l
        h = self.x*0 + h
        norm = F.conv2d(self.x, self.weight, **self.conv_args) \
            - F.conv2d(l, w_plus, **self.conv_args)*(1-gamma) \
            - F.conv2d(h, w_minus, **self.conv_args)*(1-gamma)
        R_norm = R/norm
        return self.x*F.conv_transpose2d(R_norm, self.weight, **self.conv_args) \
            - l*F.conv_transpose2d(R_norm, w_plus, **self.conv_args)*(1-gamma) \
            - h*F.conv_transpose2d(R_norm, w_minus, **self.conv_args)*(1-gamma)
    
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
        self._unpool = nn.MaxUnpool2d(kernel_size=self.kernel_size, stride=self.stride)
        
    def forward(self, x):
        self.x = x
        self.a, self.indices = nn.MaxPool2d.forward(self, x) 
        self.z = F.relu(self.a)
        return self.z
    
    def unpool(self, y):
        return self._unpool(y, self.indices, output_size=self.x.size())
    
    def zplus(self, R, *args, **kwargs):
        return self.unpool(R)
    
    def lrp_gamma(self, R, *args, **kwargs):
        return self.unpool(R)
    
    @staticmethod
    def from_module(src):
        return MaxPool(kernel_size=src.kernel_size,
                      stride=src.stride,
                      padding=src.padding,
                      dilation=src.dilation,
                      return_indices=True,
                      ceil_mode=src.ceil_mode
                      )
    
    
class ExplainableModel(LayerizedNet):
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
        super().__init__(layers)
        self.min_x, self.max_x = min_x, max_x
    
    def debug_info(self, layer, R, R_total):
        print(f"({layer.__class__.__name__}) ->", R.shape)
        print(f"({layer.__class__.__name__})     Conservation quotient: {R.sum()/R_total}")
        if R.min() < 0:
            print(f"({layer.__class__.__name__})     encountered negative values")
        print("")
    
    def deeptaylor(self, x, pattern, debug=False):
        """
        Deep Taylor Decomposition, where relevance is the outputlayer is predictions*pattern
        Pattern: eg one_hot(labels) if the heatmap for the correct class should be used
        """
        out = self.forward(x)
        R = F.relu(out*pattern)
        R_total = R.sum()
            
        for layer in self.layers[1:][::-1]:
            R = R.view(layer.z.shape)
            R = layer.zplus(R)
            if debug:
                self.debug_info(layer, R, R_total)
            
        # Select the input layer rule
        if self.min_x is None and self.max_x is None:
            R = self.layers[0].ww(R)
        elif self.min_x == 0 and self.max_x is None:
            R = self.layers[0].zplus(R)
        else:
            R = self.layers[0].zb(R, self.min_x, self.max_x)
            
        if debug:    
            self.debug_info(self.layers[0], R, R_total)
            
        return R
    
    def lrp_gamma(self, x, pattern, gamma=0.1, bias_relevance=False, debug=False):
        # Output layer relevance
        out = self.forward(x)
        R = out*pattern
        R_total = R.sum()
        
        # Backward pass
        for layer in self.layers[1:][::-1]:
            R = R.view(layer.z.shape)
            R = layer.lrp_gamma(R, gamma, bias_relevance=bias_relevance)
            if debug:
                self.debug_info(layer, R, R_total)
            
        # Input layer relevance depending on domain
        if self.min_x is None and self.max_x is None:
            R = self.layers[0].ww(R)
        elif self.min_x == 0 and self.max_x is None:
            R = self.layers[0].lrp_gamma(R, gamma)
        else:
            R = self.layers[0].zb_gamma(R, self.min_x, self.max_x, gamma=gamma)
            
        if debug:    
            self.debug_info(self.layers[0], R, R_total)
                
        return R