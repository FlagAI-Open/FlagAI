# Update Log 1.0.0

**Full Changelog**: https://github.com/OpenBMB/BMTrain/compare/0.2.3...1.0.0

## What's New

### 1. Using pytorch's hook mechanism to refactor ZeRO, checkpoint, pipeline, communication implementation

Now user can specify zero level of each `bmt.CheckpointBlock`. 

**======= Before 1.0.0 =======**

```python
import bmtrain as bmt
bmt.init_distributed(zero_level=3)

```

The zero level setting can only set globally and computation checkpointing can not be disabled. 
For `bmt.TransformerBlockList`, it has to call a blocklist forward instead of a loop way

**======= After 1.0.0 =======**

```python
import bmtrain as bmt
bmt.init_distributed()
# construct block
class Transformer(bmt.DistributedModule):
    def __init__(self,
            num_layers : int) -> None:
        super().__init__()

        self.transformers = bmt.TransformerBlockList([
            bmt.Block(
                TransformerEncoder(
                    dim_model, dim_head, num_heads, dim_ff, bias, dtype
                ), use_checkpoint=True, zero_level=3
            )
            for _ in range(num_layers)
        ])

    def forward(self):
        # return self.transformers(x) v0.2.3 can only forward in this way
        for block in self.transformers:
            x = block(x)
        return x

```

You can specify the zero level of each `bmt.CheckpointBlock` (alias of `bmt.Block`) and computation checkpointing can be disabled by setting `use_checkpoint=False` . For `bmt.TransformerBlockList`, it can be called in a loop way.


### 2. Add Bf16 support

Now BMTrain supports Bf16 training. You can simply use `dtype=torch.bfloat16' in your model construction method and BMTrain will handle the rest.

### 3. Tensor parallel implementation

For this part, BMTrain only provides a series of parallel ops for Tensor parallel implementation, including `bmt.nn.OpParallelLinear` and `bmt.nn.VPEmbedding` . We also provide a Tensor Parallel training example in our training example. You can simply use `bmt.init_distributed(tp_size=4)` to enable a 4-way tensor parallel training.

### 4. `AdamOffloadOptimizer` can save whole gathered state

Now `AdamOffloadOptimizer` can save whole gathered state. This feature can help users to save the whole gathered state of the optimizer, which can be used to resume training from the saved state. For better performance, we provide async-way save state_dict to overlap I/O and computation.
```python
import bmtrain as bmt
# you can enbale this feature in two ways: Optimmanager's or optimizer's interface
global_ckpt = bmt.optim.Optimmanager.state_dict(gather_opt=True)
global_ckpt = optimizer.state_dict(gather=True)
```
### Others

* New test for new version BMTrain