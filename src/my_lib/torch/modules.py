from functools import partial

import torch
from torch import nn as nn
from torch.nn.init import constant_
from torch.nn.init import xavier_uniform_

from my_lib.torch.activations import shifted_softplus

zeros_initializer = partial(constant_, val=0.)


class MLP(nn.Module):
    """
    Template for fully-connected neural network of the multilayer perceptron type.

    Args:
        n_in (int): Number of input nodes
        n_out (int): Number of output nodes
        n_hidden (list of int or int): Number of neurons in hidden layers.
            If this is set to an integer, it results in a rectangular layout of the network
                (i.e., same number of neuron in all hidden layers).
            If None, the MLP will have a pyramid shape
                (i.e., the number of neurons is divided by two after each layer, starting with `n_in`).
        n_layers (int): Number of layers
        activation (callable): Activation function (default: shifted softplus)
    """

    def __init__(self, n_in, n_out, n_hidden=None, n_layers=2, activation=shifted_softplus):
        super(MLP, self).__init__()
        # If no neurons are given, initialize
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = c_neurons // 2
            self.n_neurons.append(n_out)
        else:
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            self.n_neurons = [n_in] + n_hidden + [n_out]

        layers = [Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation) for i in range(n_layers - 1)]
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))

        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Network inputs.

        Returns:
            torch.Tensor: Transformed inputs

        """
        return self.out_net(inputs)


class Dense(nn.Linear):
    """ Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(self, in_features, out_features, bias=True, activation=None,
                 weight_init=xavier_uniform_, bias_init=zeros_initializer):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation

        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)

        return y
