import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

import bmtrain as bmt
from bmtrain.global_var import config
from bmtrain.distributed import all_reduce, all_gather
from .parallel_linear_func import OpParallelLinear


class VPEmbedding(bmt.DistributedModule):
    """Vocab Parallel Embedding.

    Args:
        vocab_size (int required): vocab size.
        embedding_size (int required): embedding size.
        dtype (torch.dtype): data type.
        init_mean (float optional): mean for weight init.
        init_std (float optional): std for weight init.

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__()

        self.dim_model = embedding_size
        assert vocab_size % bmt.config["tp_size"] == 0
        self.vocab_size_per_partition = vocab_size // bmt.config["tp_size"]
        self.start_index = bmt.config["tp_rank"] * self.vocab_size_per_partition
        self.end_index = (bmt.config["tp_rank"] + 1) * self.vocab_size_per_partition
        self.weight = bmt.DistributedParameter(
            torch.empty(self.vocab_size_per_partition, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(
                torch.nn.init.normal_, mean=init_mean, std=init_std
            ),
            tp_split_dim=0,
            tp_mode=True,
        )

    def forward(self, x: torch.Tensor, projection=False):
        if not projection:
            weight = all_gather(self.weight, comm=config["tp_comm"]).flatten(0, 1)
            out = F.embedding(x, weight)
            return out
        else:
            x = bmt.distributed.all_gather(x, comm=bmt.config["tp_comm"]).view(
                x.shape[0], -1, x.shape[-1]
            )
            return bmt.nn.OpParallelLinear.apply(
                x, self.weight, None, False, False, False, None, 1
            )
