# Update Log 0.2.0

## What's New

### 1. Added an `Optimizer Manager` to support various optimizer algorithms.

Before 0.2.0, the `optimizer` was strongly coupled to the "loss scaler". This results in users cannot use multiple optimizers at the same time when training model in fp16.

**======= Before 0.2.0 =======**

```python
for iteration in range(1000):
    # zero grad
    optimizer.zero_grad()

    # ...
    # loss scale and backward
    loss = optimizer.loss_scale(loss)
    loss.backward()

    # optimizer step
    bmtrain.optim_step(optimizer, lr_scheduler)
```

The `bmtrain.optim_step` allows only one `optimizer` and at most one `lr_schduler`, which cannot handle some more complex scenarios.


**======= After 0.2.0 =======**

```python
# create a new instance of optimizer manager
optim_manager = bmtrain.optim.OptimManager(loss_scale=1024)
# let optim_manager handle all the optimizer and (optional) their corresponding lr_scheduler
optim_manager.add_optimizer(optimizer, lr_scheduler)
# add_optimizer can be called multiple times to add other optimizers.

for iteration in range(1000):
    # zero grad
    optim_manager.zero_grad() # calling zero_grad for each optimizer
    
    # ...
    # loss scale and backward
    optim_manager.backward(loss)

    # optimizer step
    optim_manager.step()
```

Starting from BMTrain 0.2.0, we provide "OptimManager" to manage optimizers and loss scales. 
`OptimManager` supports managing multiple optimizers and lr_schedulers at the same time, and allows setting the loss scale independently.
`OptimManager` can also manage pytorch native optimizers, such as SGD, AdamW, etc.

### 2. Pipeline Parallelism

In this version, BMTrain has added a new kind of parallel algorithm: pipeline parallelism.
To enable pipeline parallelism, one line of code needs to be modified.

**======= ZeRO =======**
```python
layers = bmt.TransformerBlockList([
  # ...
])
```

**======= Pipeline =======**
```python
layers = bmt.PipelineTransformerBlockList([
  # ...
])
```

Replacing TransformerBlockList with PipelineTransformerBlockList allows the parallel algorithm to switch from ZeRO to pipeline parallelism.
The number of stages in the pipeline can be set by passing the `pipe_size` parameter to bmtrain.init_distributed.

### 3. Others

* Supports BF16.
* Tensors recorded in inspector supports backward propagation.
* Adds new tests.
