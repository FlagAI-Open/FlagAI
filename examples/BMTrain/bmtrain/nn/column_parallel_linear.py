import torch
from torch.nn.parameter import Parameter

import bmtrain as bmt
from bmtrain.global_var import config
from .parallel_linear_func import OpParallelLinear, ReduceType


class ColumnParallelLinear(bmt.DistributedModule):
    """Tensor Parallel use cloumn partition for Linear.

    Args:
        in_features (int): in_features size.
        out_features (int): out_features size.
        bias (bool): whether use bias.
        dtype : data type.
        gather_ouput (bool): whether gather output after compute.
        gather_input (bool): whether gather input before compute.
        async_gather_chunks (int): chunk size for async gathering data.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        gather_output=False,
        gather_input=True,
        async_gather_chunks=2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.gather_input = gather_input
        self.async_gather_chunks = async_gather_chunks
        tp_size = config["tp_size"]
        assert out_features % tp_size == 0
        self.out_features_per_partition = out_features // tp_size
        self.weight = bmt.DistributedParameter(
            torch.empty(
                self.out_features_per_partition, in_features, dtype=dtype, device="cuda"
            ),
            init_method=torch.nn.init.xavier_normal_,
            tp_split_dim=0,
            tp_mode=True,
        )
        if bias:
            self.bias = bmt.DistributedParameter(
                torch.empty(
                    self.out_features_per_partition, dtype=dtype, device="cuda"
                ),
                init_method=torch.nn.init.zeros_,
                tp_split_dim=0,
                tp_mode=True,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        gather_input = self.gather_input
        split_input = False
        reduce_output_type = None
        return OpParallelLinear.apply(
            input,
            self.weight,
            self.bias,
            gather_input,
            self.gather_output,
            split_input,
            reduce_output_type,
            self.async_gather_chunks,
        )

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features_per_partitions, self.bias is not None
        )
