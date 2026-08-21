<div align="center">

<h1><img src="docs/logo.png" height="28px" /> BMTrain</h1>

**Efficient Training for Big Models**

<p align="center">
  <a href="#overview">Overview</a> • <a href="#documentation">Documentation</a> • <a href="#install">Installation</a> • <a href="#usage">Usage</a> • <a href="#performance">Performance</a> • <a href="./README-ZH.md" target="_blank">简体中文</a>
<br>
</p>

<p align="center">

<a href='https://bmtrain.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/bmtrain/badge/?version=latest' alt='Documentation Status' />
</a>

<a href="https://github.com/OpenBMB/BMTrain/releases">
    <img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/OpenBMB/BMTrain?include_prereleases">
</a>

<a href="https://github.com/OpenBMB/BMTrain/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/BMTrain">
</a>

</p>

</div>

## What's New
- 2022/06/14 **BMTrain** [0.1.7](https://github.com/OpenBMB/BMTrain/releases/tag/0.1.7) released. ZeRO-2 optimization is supported!
- 2022/03/30 **BMTrain** [0.1.2](https://github.com/OpenBMB/BMTrain/releases/tag/0.1.2) released. Adapted to [OpenPrompt](https://github.com/thunlp/OpenPrompt)and [OpenDelta](https://github.com/thunlp/OpenDelta).
- 2022/03/16 **BMTrain** [0.1.1](https://github.com/OpenBMB/BMTrain/releases/tag/0.1.1) has publicly released the first stable version, which fixes many bugs that were in the beta version.
- 2022/02/11 **BMTrain** [0.0.15](https://github.com/OpenBMB/BMTrain/releases/tag/0.0.15) has publicly released the first beta version.

<div id="overview"></div>

## Overview

BMTrain is an efficient large model training toolkit that can be used to train large models with tens of billions of parameters. It can train models in a distributed manner while keeping the code as simple as stand-alone training.

<div id="documentation"></div>

## Documentation
Our [documentation](https://bmtrain.readthedocs.io/en/latest/index.html) provides more information about the package.

<div id="install"></div>

## Installation

- From pip （recommend） : ``pip install bmtrain``

- From source code: download the package and run ``python setup.py install``

Installing BMTrain may take a few to ten minutes, as it requires compiling the c/cuda source code at the time of installation.
We recommend compiling BMTrain directly in the training environment to avoid potential problems caused by the different environments.

<div id="usage"></div>

## Usage

### Step 1: Initialize BMTrain

Before you can use BMTrain, you need to initialize it at the beginning of your code. Just like using the distributed module of PyTorch requires the use of **init_process_group** at the beginning of the code, using BMTrain requires the use of **init_distributed** at the beginning of the code.

```python
import bmtrain as bmt
bmt.init_distributed(
    seed=0,
    zero_level=3,   # support 2 and 3 now
    # ...
)
```

**NOTE:** Do not use PyTorch's distributed module and its associated communication functions when using BMTrain.

### Step 2: Enable ZeRO Optimization

To enable ZeRO optimization, you need to make some simple replacements to the original model's code.

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

### Step 3: Enable Communication Optimization


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

### Step 4: Launch Distributed Training

BMTrain uses the same launch command as the distributed module of PyTorch.

You can choose one of them depending on your version of PyTorch.

* `${MASTER_ADDR}` means the IP address of the master node.
* `${MASTER_PORT}` means the port of the master node.
* `${NNODES}` means the total number of nodes.
* `${GPU_PER_NODE}` means the number of GPUs per node.
* `${NODE_RANK}` means the rank of this node.

#### torch.distributed.launch
```shell
$ python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node ${GPU_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} train.py
```

#### torchrun

```shell
$ torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```


For more information, please refer to the [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility).

## Example

We provide an [example](https://github.com/OpenBMB/BMTrain/tree/main/example) of training GPT-2 based on BMTrain.
The code mainly consists of the following parts.

### Part 1: Model Definition

```
├── layers
│   ├── attention.py
│   ├── embedding.py
│   ├── feedforward.py
│   ├── __init__.py
│   ├── layernorm.py
│   └── linear.py
└── models
    ├── gpt.py
    └── __init__.py
```

Above is the directory structure of the code in the part of Model Definition.

We defined all the layers needed in GPT-2 and used BMTrain's `DistributedModule` and `DistributedParameter` to enable ZeRO optimization.

### Part 2: BMTrain Initialization

```python
bmtrain.init_distributed(seed=0)

model = GPT(
    num_layers=8,
    vocab_size=10240, 
    dim_model=2560,
    dim_head=80,
    num_heads=32,
    dim_ff=8192,
    max_distance=1024,
    bias=True,
    dtype=torch.half
)

bmtrain.init_parameters(model) # or loading checkpoint use `bmtrain.load`

# ... other initialization (dataset) ...
```

`bmtrain.init_distributed(seed=0)` is used to initialize the distributed training environment and set the random seed for reproducibility.

`bmtrain.init_parameters(model)` is used to initialize the distributed parameters of the model.

### Part 3: Intialization of the Optimizer and LR Scheduler

```python
loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = bmtrain.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2)
lr_scheduler = bmtrain.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)
```

BMTrain supports *all* the PyTorch native optimizers and loss functions, and you can also use the fused optimizer provided by BMTrain for mixed-precision training.

In addition, BMTrain also provides the common LRScheduler in the `bmtrain.lr_scheduler` module.

### Part 4: Training Loop

```python
# create a new instance of optimizer manager
optim_manager = bmtrain.optim.OptimManager(loss_scale=1024)
# let optim_manager handle all the optimizer and (optional) their corresponding lr_scheduler
optim_manager.add_optimizer(optimizer, lr_scheduler)
# add_optimizer can be called multiple times to add other optimizers.

for iteration in range(1000):
    # ... load data for each rank ...

    # forward pass and calculate loss
    pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)
    logits = model(
        enc_input,
        pos,
        pos < enc_length[:, None]
    )
    batch, seq_len, vocab_out_size = logits.size()

    loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
    
    global_loss = bmtrain.sum_loss(loss).item() # sum the loss across all ranks. This is only used for the training log

    # zero grad
    optim_manager.zero_grad() # calling zero_grad for each optimizer

    # loss scale and backward
    optim_manager.backward()

    # clip grad norm
    grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=1.0)

    # optimizer step
    optim_manager.step()

    # ... save checkpoint or print logs ...
```

The training loop part will be slightly longer, but just like a normal training loop, you don't need to adapt much to distributed training.

You can follow the comments in the code to get an idea of what each section of code is doing.

The only additional note is `optimizer`. After using BMTrain, some details in optimizers should be adjusted. We have implemented all those details needed in `optim_manager`. What you need is just letting `optim_manager` to handle all the optimizers by `add_optimizer`, and letting `optim_manager` do `zero_grad()`, `backward()`, `clip_grad_norm()` and `step()` instead.

If you are not using the mixed-precision training, you can train without `loss_scale`. Just set `loss_scale` to None in the `__init__` function of `OptimManager(loss_scale=None)`, which is also the default.

If you are using mixed-precision training, *loss scale* is the technique widely used in mixed precision training to prevent gradient underflow. By using `optim_manager.backward(loss)` to scale the `loss` before backward and set `loss_scale` to some floating number in the `__init__` function of `OptimManager`。The `loss_scale` would be adjusted adaptively based on the gradient during training.

<div id="performance"></div>

## Performance

We trained a GPT-2 model with 13B parameters using 4 servers with 8 V100s on each server, and measured the throughput of each GPU during the training process (samples per GPU per second).

Model structure:
* 40 layers
* 128 attention heads
* 5120 hidden dimension
* 512 sequence length


| batch size  | 8     | 16    | 24    | 32    |
|-------------|-------|-------|:------|:------|
| BMTrain     | 24.15 | 26.94 | 29.42 | 28.28 |
| ZeRO3(mp=1) | 14.88 | 21.69 | 24.38 | -     |
| ZeRO3(mp=4) | 15.51 | -     | -     | -     |
| ZeRO3(mp=8) | 15.51 | -     | -     | -     |
| ZeRO2(mp=1) | -     | -     | -     | -     |
| ZeRO2(mp=4) | 22.85 | -     | -     | -     |
| ZeRO2(mp=8) | 21.33 | -     | -     | -     |

**ZeROa(mp=b)** means DeepSpeed + Megatron ZeRO stage a and model parallelism = b.

**-** in the table means out of memory.

## Supported Models

We have migrated most of the common models in NLP to the BMTrain. You can find the list of supported models in the repo [ModelCenter](https://github.com/OpenBMB/ModelCenter).

## Community
We welcome everyone to contribute codes following our [contributing guidelines](https://github.com/OpenBMB/BMTrain/blob/master/CONTRIBUTING.md).

You can also find us on other platforms:
- QQ Group: 735930538
- Website: https://www.openbmb.org
- Weibo: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## License
The package is released under the [Apache 2.0](https://github.com/OpenBMB/BMTrain/blob/master/LICENSE) License.

## Other Notes

`BMTrain` makes underlying changes to PyTorch, so if your program outputs unexpected results, you can submit information about it in an issue.

