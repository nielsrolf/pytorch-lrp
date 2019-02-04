import copy
from trainable_net import TrainableNet


class Layer():
    """
    Base class for all layers
    """
    @classmethod
    def from_module(cls, src):
        layer = copy.deepcopy(src)
        layer.__class__ = cls
        return layer
    

class LayerizedNet(TrainableNet):
    def __init__(self, layers):
        """
        Layers: Instances of subclasses of Layer
        """
        super().__init__()
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    @classmethod
    def from_model(cls, model, layer_names, layer_base_type, last_layer_type, *args, **kwargs):
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
        layer_base_type: looks for layer classes that subclass this type to convert the model
        last_layer_type: type of the last layer
        *args, **kwargs: will be passed to the constructor of `cls`, i.e. LayerizedNet or a subclass
        
        Nonlinearities don't have to be specified if they are implemented in layer_base_type.
        Alternatively, they can be treated as extra layers.
        Reshaping does not have to be specified
        
        Relies on the fact that each ReluLayer is also subclass of the class of model
        """
        def casted_layer(src):
            # finds the layer equivalent of src.__class__
            for layer_type in layer_base_type.__subclasses__():
                if issubclass(layer_type, src.__class__):
                    return layer_type.from_module(src)
        
        # cast all layers
        # all but the last are casted to relu layers
        layers = [casted_layer(model._modules[l]) for l in layer_names[:-1]] + \
                 [last_layer_type.from_module(model._modules[layer_names[-1]])]
        
        # now create a ExplainBase object which we call self, and pretend to be a constructor
        self = cls(layers, *args, **kwargs)
        # add the layers to the properties of self, in hope that 
        # pytorch transforms it to a self._modules element
        for l, layer in zip(layer_names, layers):
            self.__dict__[l] = layer
            
        return self