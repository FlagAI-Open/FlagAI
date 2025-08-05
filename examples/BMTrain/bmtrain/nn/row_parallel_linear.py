import torch
from torch.nn.parameter import Parameter

import bmtrain as bmt
from bmtrain.global_var import config
from .parallel_linear_func import OpParallelLinear, ReduceType


class RowParallelLinear(bmt.DistributedModule):
    """Tensor Parallel use row partition for Linear.

    Args:
        in_features (int): in_features size.
        out_features (int): out_features size.
        bias (bool): whether use bias.
        dtype : data type.
        split_input (bool): whether split input before compute.
        all_reduce_output (bool): if true use all_reduce data after compute, or use reduce_scatter.
        async_chunks (int): chunk size for async.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        split_input=False,
        all_reduce_output=False,
        async_chunks=2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_input = split_input
        self.all_reduce_output = all_reduce_output
        self.async_chunks = async_chunks
        tp_size = config["tp_size"]
        assert in_features % tp_size == 0
        self.in_features_per_partition = in_features // tp_size
        self.weight = bmt.DistributedParameter(
            torch.empty(
                self.out_features,
                self.in_features_per_partition,
                dtype=dtype,
                device="cuda",
            ),
            init_method=torch.nn.init.xavier_normal_,
            tp_split_dim=1,
            tp_mode=True,
        )
        if bias:
            self.bias = bmt.DistributedParameter(
                torch.empty(self.out_features, dtype=dtype, device="cuda"),
                init_method=torch.nn.init.zeros_,
                tp_split_dim=-1,
                tp_mode=True,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        gather_input = self.split_input
        gather_output = False
        reduce_output_type = (
            ReduceType.ALL_REDUCE
            if self.all_reduce_output
            else ReduceType.REDUCE_SCATTER
        )
        out = OpParallelLinear.apply(
            input,
            self.weight,
            None,
            gather_input,
            gather_output,
            self.split_input,
            reduce_output_type,
            self.async_chunks,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features_per_partition, self.out_features, self.bias is not None
        )
