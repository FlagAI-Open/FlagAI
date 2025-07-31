import torch
import bmtrain as bmt
import torch.nn.functional as F
import collections
from itertools import repeat
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
class Identity(bmt.DistributedModule):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, input):
        return input
class Conv2d(bmt.DistributedModule):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                dtype=torch.float,
                int8: bool=False,
                init_mean : float=0.0,
                init_std : float = 1,
                bias : bool=True,
                padding_mode='zeros',
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.transposed = None
        self.output_padding = None

        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.padding_mode = padding_mode

        kernel = to_2tuple(kernel_size)
        self.weight = bmt.DistributedParameter(
            torch.empty((out_channels, int(in_channels/groups), kernel[0], kernel[1]), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((out_channels,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
        ) if bias else None
        self.int8=int8
    def forward(self, x : torch.Tensor):
        x = F.conv2d(x,
                    weight=self.weight, 
                    bias=self.bias, 
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    )
        
        return x