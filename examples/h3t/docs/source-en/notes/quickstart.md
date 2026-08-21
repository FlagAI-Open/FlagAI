# Quick Start

## Step 1: Initialize BMTrain

Before you can use BMTrain, you need to initialize it at the beginning of your code. Just like using the distributed module of PyTorch requires the use of **init_process_group** at the beginning of the code, using BMTrain requires the use of **init_distributed** at the beginning of the code.

```python
import bmtrain as bmt
bmt.init_distributed(
    seed=0,
    # ...
)
```

**NOTE:** Do not use PyTorch's distributed module and its associated communication functions when using BMTrain.

## Step 2: Enable ZeRO-3 Optimization

To enable ZeRO-3 optimization, you need to make some simple replacements to the original model's code.

* `torch.nn.Module` -> `bmtrain.DistributedModule`
* `torch.nn.Parameter` -> `bmtrain.DistributedParameter`

And wrap the transformer blocks with `bmtrain.CheckpointBlock`.

Here is an example.

**Original**

```python
import torch
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(1024))
        self.module_list = torch.nn.ModuleList([
            SomeTransformerBlock(),
            SomeTransformerBlock(),
            SomeTransformerBlock()
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x, 1, 2, 3)
        return x

```

**Replaced**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule): # changed here
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024)) # changed here
        self.module_list = torch.nn.ModuleList([
            bmt.CheckpointBlock(SomeTransformerBlock()), # changed here
            bmt.CheckpointBlock(SomeTransformerBlock()), # changed here
            bmt.CheckpointBlock(SomeTransformerBlock())  # changed here
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x, 1, 2, 3)
        return x
    
```

## Step 3: Enable Communication Optimization


To further reduce the extra overhead of communication and overlap communication with computing time, `TransformerBlockList` can be used for optimization.

You can enable them by making the following substitutions to the code:

* `torch.nn.ModuleList` -> `bmtrain.TransformerBlockList`
* `for module in self.module_list: x = module(x, ...)` -> `x = self.module_list(x, ...)`

**Original**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024))
        self.module_list = torch.nn.ModuleList([
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock())
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x, 1, 2, 3)
        return x
    
```

**Replaced**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024))
        self.module_list = bmt.TransformerBlockList([ # changed here
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock())
        ])
    
    def forward(self):
        x = self.param
        x = self.module_list(x, 1, 2, 3) # changed here
        return x
    
```

## Step 4: Launch Distributed Training

BMTrain uses the same launch command as the distributed module of PyTorch.

You can choose one of them depending on your version of PyTorch.

* `${MASTER_ADDR}` means the IP address of the master node.
* `${MASTER_PORT}` means the port of the master node.
* `${NNODES}` means the total number of nodes.
* `${GPU_PER_NODE}` means the number of GPUs per node.
* `${NODE_RANK}` means the rank of this node.

### torch.distributed.launch
```shell
$ python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node ${GPU_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} train.py
```

### torchrun

```shell
$ torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```


For more information, please refer to the [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility).

## Other Notes

`BMTrain` makes underlying changes to PyTorch, so if your program outputs unexpected results, you can submit information about it in an issue.

For more examples, please refer to the *examples* folder.

