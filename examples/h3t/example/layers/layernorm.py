from typing import Tuple
import torch
import torch.nn.functional as F
import bmtrain as bmt

class Layernorm(bmt.DistributedModule):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                dtype=None) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = bmt.DistributedParameter(torch.empty(self.normalized_shape, dtype=dtype, device="cuda"), init_method=torch.nn.init.ones_)
            self.bias = bmt.DistributedParameter(torch.empty(self.normalized_shape, dtype=dtype, device="cuda"), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
