import torch
import math
import torch.nn as nn

import torch.nn as nn
from transformers.activations import get_activation

class Activations(nn.Module):
    """
    Implementation of various activation function. Copied from open-source project AdapterHub #TODO: addlink
    """

    def __init__(self, activation_type):
        self.activation_type = activation_type
        if activation_type.lower() == "relu":
            self.f = nn.functional.relu
        elif activation_type.lower() == "tanh":
            self.f = torch.tanh
        elif activation_type.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif activation_type.lower() == "gelu_new":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif activation_type.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif activation_type.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(activation_type)

        super().__init__()

    def forward(self, x):
        return self.f(x)
    
    def __repr__(self):
        return self.activation_type


