# Turn model into Megatron-LM version


Most of the process in `parallel` are taken from `Megatron-LM`，and is put in `mpu`module

## 1.Turn MLP layer into column/rowParallel version
Location：`flagai/model/layers/embeddings_mpu.py`

Key idea:
split the two forward1 layers in 1linear1 layer, following the column-fist principle
```python
intermediate_parallel = self.dense_h_to_4h(hidden_states)
intermediate_parallel = gelu(intermediate_parallel)
output = self.dense_4h_to_h(intermediate_parallel)
```
`self.dense_h_to_4h` and`self.dense_4h_to_h`：
```python
>>>      # Project to 4h.
>>>     self.dense_h_to_4h = ColumnParallelLinear(hidden_size,
>>>                                               4 * hidden_size,
>>>                                               gather_output=False, #这里可以是True
>>>                                               init_method=init_method)
>>>     # Project back to h.
>>>     self.dense_4h_to_h = RowParallelLinear(
>>>         4 * hidden_size,
>>>         hidden_size,
>>>         input_is_parallel=True,# 受到self.dense_h_to_4h的gather_output设置影响
>>>         init_method=output_layer_init_method)
```
Key parameters:
```python
>>> world_size = get_model_parallel_world_size()
>>> self.hidden_size_per_partition = divide(hidden_size, world_size)
>>> self.hidden_size_per_attention_head = divide(hidden_size,
>>>                                              num_attention_heads)
>>> self.num_attention_heads_per_partition = divide(
>>>     num_attention_heads, world_size)
```
Here is the code for code of two `parallel linear`

## 2. Transform `self-Attention`
Turn  two Linear layer in `self-Attention`into`column/rowParallel` version. Location：`flagai/model/layers/attentions_mpu.py`
### 2.1 Turn the linear layer that projects`x` to  `q, k, v`  into `collumnParalleLinear`
```python
>>> self.query_key_value = ColumnParallelLinear(hidden_size,
>>>                                                 3 * hidden_size,
>>>                                                 stride=3,
>>>                                                 gather_output=False,
>>>                                         init_method=init_method)
```
### 2.2 Turn the output`dense`layer into`rowparallelLinear`
```python
>>> self.dense = RowParallelLinear(hidden_size,
>>>                                hidden_size,
>>>                                input_is_parallel=True,
>>>                                init_method=output_layer_init_method)
```
## 3.Transform two `Linear Layer` in `cross-Attention` into `column/rowParallel` version as above
```python
>>> self.query = ColumnParallelLinear(hidden_size,
>>>                                   hidden_size,
>>>                                   gather_output=False,
>>>                                   init_method=init_method)
>>> self.key_value = ColumnParallelLinear(hidden_size,
>>>                                       2 * hidden_size,
>>>                                       stride=2,
>>>                                       gather_output=False,
>>>                                       init_method=init_method)
>>> # Output.
>>> self.dense = RowParallelLinear(hidden_size,
>>>                                hidden_size,
>>>                                input_is_parallel=True,
>>>                                init_method=output_layer_init_method)
>>> 
```
