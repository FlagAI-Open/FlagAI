# 转化一个模型为Megatron-LM的模型并行版本
- [转化model为Megatron-LM版本](#转化model为megatron-lm版本)
  - [1.将MLP层转化为column/rowParallel版本](#1将mlp层转化为columnrowparallel版本)
  - [2.转化`self-Attention`](#2转化self-attention)
    - [2.1 将`x` proj 到  `q, k, v` 的`linear `层，转化为 `columnParalleLinear`](#21-将x-proj-到--q-k-v-的linear-层转化为-columnparallelinear)
    - [2.2 将输出的`dense`层转化为`rowparallelLinear`](#22-将输出的dense层转化为rowparallellinear)
  - [3.将`cross-Attention`中的两个`Linear Layer `转化为`column/rowParallel`版本](#3将cross-attention中的两个linear-layer-转化为columnrowparallel版本)

飞智支持了BERT，GLM，GPT2 以及T5模型的megatron-lm模型并行. 其主要的操作都是从`Megatron-LM`中拿出来的，放在了`mpu`模块中。

## 1.将MLP层转化为column/rowParallel版本
代码位置：`flagai/model/layers/embeddings_mpu.py` 
核心思想:将两个`linear`层的`forward`过程，按照特定顺序进行拆分（先列后行）：
```python
intermediate_parallel = self.dense_h_to_4h(hidden_states)
intermediate_parallel = gelu(intermediate_parallel)
output = self.dense_4h_to_h(intermediate_parallel)
```
其中，`self.dense_h_to_4h` 和`self.dense_4h_to_h` 分别为：
```python
     # Project to 4h.
    self.dense_h_to_4h = ColumnParallelLinear(hidden_size,
                                              4 * hidden_size,
                                              gather_output=False, #这里可以是True
                                              init_method=init_method)
    # Project back to h.
    self.dense_4h_to_h = RowParallelLinear(
        4 * hidden_size,
        hidden_size,
        input_is_parallel=True,# 受到self.dense_h_to_4h的gather_output设置影响
        init_method=output_layer_init_method)
```
关键参数
```python
world_size = get_model_parallel_world_size()
self.hidden_size_per_partition = divide(hidden_size, world_size)
self.hidden_size_per_attention_head = divide(hidden_size,
                                             num_attention_heads)
self.num_attention_heads_per_partition = divide(
    num_attention_heads, world_size)
```

## 2.转化`self-Attention` 
`self-attention`中的两个Linear layer分别转化为`column/rowParallel` 版本

代码位置：`flagai/model/layers/attentions_mpu.py`
### 2.1 将`x` proj 到  `q, k, v` 的`linear `层，转化为 `columnParalleLinear`
### 2.2 将输出的`dense`层转化为`rowparallelLinear`
如下：
```python
self.query_key_value = ColumnParallelLinear(hidden_size,
                                                3 * hidden_size,
                                                stride=3,
                                                gather_output=False,
                                        init_method=init_method)
if relative_encoding:
    self.relative = ColumnParallelLinear(hidden_size,
                                         hidden_size,
                                         gather_output=False,
                                         init_method=init_method)
# Dropout. Note that for a single iteration, this layer will generate
# different outputs on different number of parallel partitions but
# on average it should not be partition dependent.
self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

# Output.
self.dense = RowParallelLinear(hidden_size,
                               hidden_size,
                               input_is_parallel=True,
                               init_method=output_layer_init_method)
```
## 3.将`cross-Attention`中的两个`Linear Layer `转化为`column/rowParallel`版本
同上，
```python
self.query = ColumnParallelLinear(hidden_size,
                                  hidden_size,
                                  gather_output=False,
                                  init_method=init_method)
self.key_value = ColumnParallelLinear(hidden_size,
                                      2 * hidden_size,
                                      stride=2,
                                      gather_output=False,
                                      init_method=init_method)
# Dropout. Note that for a single iteration, this layer will generate
# different outputs on different number of parallel partitions but
# on average it should not be partition dependent.
self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

# Output.
self.dense = RowParallelLinear(hidden_size,
                               hidden_size,
                               input_is_parallel=True,
                               init_method=output_layer_init_method)

```
