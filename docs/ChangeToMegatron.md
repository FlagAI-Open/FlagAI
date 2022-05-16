# Turn model into Megatron-LM version
Three steps：
- [1] Turn`MLP`layer into`column/rowParallel` version
- [2] Turn two `Linear layer` in `self-Attention` into `column/rowParallel `version
- [3] Turn two `Linear Layer` in `cross-Attention`into `column/rowParallel`version

Most of the process in `parallel` are taken from `Megatron-LM`，and is put in `mpu`module

## 1.Turn MLP layer into column/rowParallel版本
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
Key parameters:
```python
world_size = get_model_parallel_world_size()
self.hidden_size_per_partition = divide(hidden_size, world_size)
self.hidden_size_per_attention_head = divide(hidden_size,
                                             num_attention_heads)
self.num_attention_heads_per_partition = divide(
    num_attention_heads, world_size)
```
Here is the code for code of two `parallel linear`
### ColumnParallelLinear
```python

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 gather_output=True,
                 init_method=init.xavier_normal_,
                 stride=1,
                 keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(
            torch.Tensor(self.output_size_per_partition, self.input_size))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            self.bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.output_size,
            self.input_size,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
            # gather_from_model_parallel_region 是mpu中提供的功能
        else:
            output = output_parallel
        return output
```
### RowParallelLinear
```python
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_,
                 stride=1,
                 keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(
            torch.Tensor(self.output_size, self.input_size_per_partition))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.output_size,
            self.input_size,
            self.input_size_per_partition,
            1,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        # reduce_from_model_parallel_region 是mpu提供的功能
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
```
## 2.Turn  two Linear layerin `self-Attention`into`column/rowParallel` version
Location：`flagai/model/layers/attentions_mpu.py`
### 2.1 Turn the linear layer that projects`x` to  `q, k, v`  into `collumnParalleLinear`
### 2.2 Turn the output`dense`layer into`rowparallelLinear`
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
## 3.Turn two `Linear Layer ` in `cross-Attention`into`column/rowParallel`version
as above
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
