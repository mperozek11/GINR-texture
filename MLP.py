import numpy as np
import torch
from torch import nn

class Sine(nn.Module):
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        return torch.sin(self.omega_0 * input)
    
class MLP(nn.Module):
    """
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        hidden_dim: int = 512, number of neurons in hidden layers
        n_layers: int = 4, number of layers (total, including first and last)
        geometric_init: bool = False, initialize weights so that output is spherical
        beta: int = 0, if positive, use SoftPlus(beta) instead of ReLU activations
        sine: bool = False, use SIREN activation in the first layer
        all_sine: bool = False, use SIREN activations in all other layers
        skip: bool = True, add a skip connection to the middle layer
        bn: bool = False, use batch normalization
        dropout: float = 0.0, dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        geometric_init: bool,
        beta: int,
        sine: bool,
        all_sine: bool,
        skip: bool,
        bn: bool,
        dropout: float,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = all_sine
        self.skip = skip
        self.bn = bn
        self.dropout = dropout

        # Modules
        self.model = nn.ModuleList()
        in_dim = input_dim
        out_dim = hidden_dim
        for i in range(n_layers):
            layer = nn.Linear(in_dim, out_dim)

            # Custom initializations
            if geometric_init:
                if i == n_layers - 1:
                    geometric_initializer(layer, in_dim)
            elif sine:
                if i == 0:
                    first_layer_sine_initializer(layer)
                elif all_sine:
                    sine_initializer(layer)

            self.model.append(layer)

            # Activation, BN, and dropout
            if i < n_layers - 1:
                if sine:
                    if i == 0:
                        act = Sine()
                    else:
                        act = Sine() if all_sine else nn.Tanh()
                
                elif beta > 0:
                    act = nn.Softplus(beta=beta)  # IGR uses Softplus with beta=100
                else:
                    act = nn.ReLU(inplace=True)
                self.model.append(act)
                if bn:
                    self.model.append(nn.LayerNorm(out_dim))
                if dropout > 0:
                    self.model.append(nn.Dropout(dropout))

            in_dim = hidden_dim
            # Skip connection
            if i + 1 == int(np.ceil(n_layers / 2)) and skip:
                self.skip_at = len(self.model)
                in_dim += input_dim

            out_dim = hidden_dim
            if i + 1 == n_layers - 1:
                out_dim = output_dim

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.model):
            if i == self.skip_at:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)

        return x
    


def geometric_initializer(layer, in_dim):
    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.00001)
    nn.init.constant_(layer.bias, -1)


def first_layer_sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, "weight"):
            num_input = layer.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(-1 / num_input, 1 / num_input)


def sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, "weight"):
            num_input = layer.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30
            )