# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# feedforward
import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from .layer_norm import T5LayerNorm
from .activations import gelu_impl, relu, gelu_new, ACT2FN
import torch.nn.functional as F
from flagai.mpu.initialize import get_model_parallel_rank
from flagai.mpu.initialize import get_model_parallel_world_size
from flagai.mpu.mappings import copy_to_model_parallel_region
from flagai.mpu.mappings import gather_from_model_parallel_region
from flagai.mpu.mappings import reduce_from_model_parallel_region
from flagai.mpu.mappings import scatter_to_model_parallel_region
from flagai.mpu.utils import divide

from flagai.model.layers.linear import CPM3Linear

from .layer_norm import CPM3LayerNorm


def _initialize_affine_weight(weight,
                              output_size,
                              input_size,
                              per_partition_size,
                              partition_dim,
                              init_method,
                              stride=1,
                              return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # If we only use 1 process for model parallelism, bypass scatter.
    if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
        world_size = get_model_parallel_world_size()
    else:
        world_size = 1
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(output_size,
                                input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight,
                              per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class GPT2MLP(nn.Module):

    def __init__(self,
                 n_state,
                 config,
                 init_method=init.xavier_normal_
                 ):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.hidden_size
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            self.c_fc = ColumnParallelLinear(nx,
                                             n_state,
                                             stride=1,
                                             gather_output=False,
                                             init_method=init_method)

            self.c_proj = RowParallelLinear(input_size=n_state,
                                            output_size=nx,
                                            bias=True,
                                            input_is_parallel=True,
                                            stride=1,
                                            init_method=init_method)
        else:
            self.c_fc = nn.Linear(nx, n_state)
            self.c_proj = nn.Linear(n_state, nx)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class T5DenseReluDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            self.wi = ColumnParallelLinear(config['d_model'],
                                           config['d_ff'],
                                           bias=False,
                                           gather_output=False)
            self.wo = RowParallelLinear(config['d_model'],
                                        config['d_ff'],
                                        bias=False,
                                        input_is_parallel=True)
        else:
            self.wi = nn.Linear(config['d_model'], config['d_ff'], bias=False)
            self.wo = nn.Linear(config['d_ff'], config['d_model'], bias=False)
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = relu(hidden_states)
        hidden_states = self.dropout(
            hidden_states)  # different from MLPForward
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            self.wi_0 = ColumnParallelLinear(config['d_model'],
                                             config['d_ff'],
                                             bias=False,
                                             gather_output=False)

            self.wi_1 = ColumnParallelLinear(config['d_model'],
                                             config['d_ff'],
                                             bias=False,
                                             gather_output=False)
            self.wo = RowParallelLinear(config['d_ff'],
                                        config['d_model'],
                                        bias=False,
                                        input_is_parallel=True)
        else:
            self.wi_0 = nn.Linear(config['d_model'],
                                  config['d_ff'],
                                  bias=False)
            self.wi_1 = nn.Linear(config['d_model'],
                                  config['d_ff'],
                                  bias=False)
            self.wo = nn.Linear(config['d_ff'], config['d_model'], bias=False)

        self.dropout = nn.Dropout(config['dropout_rate'])
        self.gelu_act = gelu_new

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config['feed_forward_proj'] == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config['feed_forward_proj'] == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config['feed_forward_proj']} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config['d_model'],
                                      eps=config['layer_norm_epsilon'])
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class MLPForward(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        output_dropout_prob,
        init_method,
        output_layer_init_method=None,
    ):
        super(MLPForward, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            self.dense_h_to_4h = ColumnParallelLinear(hidden_size,
                                                      4 * hidden_size,
                                                      gather_output=False,
                                                      init_method=init_method)
            # Project back to h.
            self.dense_4h_to_h = RowParallelLinear(
                4 * hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
        else:
            self.dense_h_to_4h = nn.Linear(
                hidden_size,
                4 * hidden_size,
            )
            # Project back to h.
            self.dense_4h_to_h = nn.Linear(
                4 * hidden_size,
                hidden_size,
            )
        self.dropout = nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, **kw_args):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu_impl(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class BertPooler(nn.Module):

    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


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
        else:
            output = output_parallel
        return output


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
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


class DenseGatedACT(torch.nn.Module):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 0.02,
                 bias = False,
                 length_scale : bool = False,
        ):
        super().__init__()

        self.w_0 = CPM3Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            bias = bias,
        )

        self.w_1 = CPM3Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            bias = bias,
        )

        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))
    
    def forward(self, x : torch.Tensor):
        """ This model inherits from bmt.DistributedModule. 
            Transform an input tensor from one feature space to another via a nonlinear operation
        
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``) 

        """
        gelu_score = self.act( self.w_0(x) )
        x = self.w_1(x)

        x = gelu_score * x
        return x


class DenseACT(torch.nn.Module):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 0.02,
                 bias = False,
                 length_scale : bool = False,
        ):
        super().__init__()

        self.w = CPM3Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            bias = bias,
        )
        
        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))

    def forward(self, x : torch.Tensor):
        """ This model inherits from bmt.DistributedModule. 
            Transform an input tensor from one feature space to another via a nonlinear operation
        
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``) 
        """
        x = self.w(x)
        x = self.act(x)
        
        return x


class FeedForward(torch.nn.Module):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """

    def __init__(self,
                 dim_in : int, 
                 dim_ff : int,
                 dim_out : int = None,
                 dtype = torch.half, 
                 int8 = False,
                 init_mean = 0.0, 
                 init_std = 0.02,
                 bias = False,
                 activate_fn = "gated_gelu",
                 length_scale : bool = False,
                 dropout_p = 0,
        ):

        super().__init__()

        if activate_fn.startswith("gated_"):
            self.w_in = DenseGatedACT(
                dim_in = dim_in,
                dim_ff = dim_ff,
                activate_fn = activate_fn[6:],
                dtype = dtype,
                int8 = int8,
                init_mean = init_mean,
                init_std = init_std,
                bias = bias,
                length_scale = length_scale,
            )
        else:
            self.w_in = DenseACT(
                dim_in = dim_in,
                dim_ff = dim_ff,
                activate_fn = activate_fn,
                dtype = dtype,
                int8 = int8,
                init_mean = init_mean,
                init_std = init_std,
                bias = bias,
                length_scale = length_scale,
            )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        if dim_out is None:
            dim_out = dim_in

        self.dim_ff = dim_ff
        self.dim_out = dim_out

        self.w_out = CPM3Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = True,
            dtype = dtype,
            int8 = int8,
            bias = bias,
        )

        self.int8 = int8
        self.length_scale = length_scale

    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)

        return x


class CPM3FFN(torch.nn.Module):
    """ The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, 
                 dim_model : int, 
                 dim_ff : int,
                 dtype = torch.half, 
                 int8 = False,
                 norm_init_var : float = 1.0,
                 norm_bias : bool = False,
                 norm_eps : float = 1e-5, 
                 ffn_init_mean : float = 0.0, 
                 ffn_init_std : float = 0.02,
                 ffn_bias : bool = False,
                 ffn_activate_fn : str = "gated_gelu",
                 post_layer_norm : bool = False,
                 length_scale : bool = False,
                 dropout_p : float = 0,
                ):
        super().__init__()

        self.layernorm_before_ffn = CPM3LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        self.ffn = FeedForward(
            dim_in = dim_model, 
            dim_ff = dim_ff, 
            dim_out = dim_model, 
            dtype = dtype, 
            int8 = int8,
            init_mean = ffn_init_mean, 
            init_std = ffn_init_std,
            bias = ffn_bias,
            activate_fn = ffn_activate_fn,
            length_scale = length_scale,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,
               ):
        """ 
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """ 
        x = self.layernorm_before_ffn(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states