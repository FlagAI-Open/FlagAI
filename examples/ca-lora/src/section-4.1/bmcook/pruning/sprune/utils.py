import types
import torch
from torch import Tensor
from torch.nn import Module
from typing import List, Dict, Union, Optional, NewType

SPlugin = NewType('Plugin', Dict[str, Optional[Union[int, Tensor]]])

################################# utils for forward-inside pruning setup #################################
def set_pruning_transformer(module: Module, index: int, TRANSFORMER_MASK: List[SPlugin], is_bmtCBlock: bool = True):
    module.index = index
    if is_bmtCBlock:
        module.forward_unprune = module._module.forward
    else:
        module.forward_unprune = module.forward
    def prune_forward(module_self, self_hidden_states, *args, **kwargs):
        index = module_self.index
        mask = TRANSFORMER_MASK[index]['mask']
        out = module_self.forward_unprune(self_hidden_states, *args, **kwargs)
        if mask is not None:
            out = self_hidden_states + (out - self_hidden_states) * mask
        return out
    if is_bmtCBlock:
        module._module.forward = types.MethodType(prune_forward, module)
    else:
        module.forward = types.MethodType(prune_forward, module)

def set_pruning_att(module: Module, index: int, ATTENTION_MASK: List[SPlugin], NUM_HEADS_MASK: List[SPlugin], DIM_HEAD_MASK: List[SPlugin]):
    module.index = index
    module.forward_unprune = module.forward
    def prune_forward(module_self, hidden_states, *args, **kwargs):
        index = module_self.index
        mask = ATTENTION_MASK[index]['mask']
        out = module_self.forward_unprune(hidden_states, *args, **kwargs)
        if mask is not None:
            out = hidden_states + (out - hidden_states) * mask
        return out
    module.forward = types.MethodType(prune_forward, module)
    # prune Linears:
    for s_name, s_module in module.named_modules():
        if 'project' in s_name:
            set_pruning_linear_attention(s_module, index, NUM_HEADS_MASK, DIM_HEAD_MASK, 'in')
        elif 'attention_out' in s_name:
            set_pruning_linear_attention(s_module, index, NUM_HEADS_MASK, DIM_HEAD_MASK, 'out')

def set_pruning_ffn(module: Module, index: int, FFN_MASK: List[SPlugin], DIM_FF_MASK: List[SPlugin]):
    module.index = index
    module.forward_unprune = module.forward
    def prune_forward(module_self, hidden_states, *args, **kwargs):
        index = module_self.index
        mask = FFN_MASK[index]['mask']
        out = module_self.forward_unprune(hidden_states, *args, **kwargs)
        if mask is not None:
            out = hidden_states + (out - hidden_states) * mask
        return out
    module.forward = types.MethodType(prune_forward, module)
    # prune Linears
    for s_name, s_module in module.named_modules():
        if 'w_in.w' in s_name:
            set_pruning_linear_feedforward(s_module, index, DIM_FF_MASK, 'in')
        elif 'w_out' in s_name:
            set_pruning_linear_feedforward(s_module, index, DIM_FF_MASK, 'out')

def set_pruning_linear_attention(module: Module, index: int, NUM_HEADS_MASK: List[SPlugin], DIM_HEAD_MASK: List[SPlugin], in_out: str):
    assert len(NUM_HEADS_MASK) == len(DIM_HEAD_MASK)
    module.index = index
    module.forward_unprune = module.forward
    if in_out == 'in':
        def prune_forward(module_self, *args, **kwargs):
            index = module_self.index
            num_heads, dim_head = NUM_HEADS_MASK[index]['dim'], DIM_HEAD_MASK[index]['dim']
            num_heads_mask, dim_head_mask = NUM_HEADS_MASK[index]['mask'], DIM_HEAD_MASK[index]['mask']
            out = module_self.forward_unprune(*args, **kwargs)  # (batch, len, num_heads*dim_head)
            if num_heads_mask is not None:
                old_size = out.size()
                out = out.view(old_size[0], old_size[1], num_heads, dim_head)
                out = out * num_heads_mask.view(num_heads, 1)
                out = out.view(old_size[0], old_size[1], num_heads * dim_head)
            if dim_head_mask is not None:
                old_size = out.size()
                out = out.view(old_size[0], old_size[1], num_heads, dim_head)
                out = out * dim_head_mask
                out = out.view(old_size[0], old_size[1], num_heads * dim_head)
            return out
    elif in_out == 'out':
        def prune_forward(module_self, x, *args, **kwargs):
            index = module_self.index
            num_heads, dim_head = NUM_HEADS_MASK[index]['dim'], DIM_HEAD_MASK[index]['dim']
            num_heads_mask, dim_head_mask = NUM_HEADS_MASK[index]['mask'], DIM_HEAD_MASK[index]['mask']
            if num_heads_mask is not None:
                old_size = x.size()  # (batch, len, num_heads * dim_head)
                x = x.view(old_size[0], old_size[1], num_heads, dim_head)
                x = x * num_heads_mask.view(num_heads, 1)
                x = x.view(old_size[0], old_size[1], num_heads * dim_head)
            if dim_head_mask is not None:
                old_size = x.size()
                x = x.view(old_size[0], old_size[1], num_heads, dim_head)
                x = x * dim_head_mask
                x = x.view(old_size[0], old_size[1], num_heads * dim_head)
            out = module_self.forward_unprune(x, *args, **kwargs)  # (batch, len, dim_model)
            return out
    module.forward = types.MethodType(prune_forward, module)

def set_pruning_linear_feedforward(module: Module, index: int, DIM_FF_MASK: List[SPlugin], in_out: str):
    module.index = index
    module.forward_unprune = module.forward
    if in_out == 'in':
        def prune_forward(module_self, *args, **kwargs):
            index = module_self.index
            mask = DIM_FF_MASK[index]['mask']
            out = module_self.forward_unprune(*args, **kwargs)
            if mask is not None:
                out = out * mask
            return out
    elif in_out == 'out':
        def prune_forward(module_self, x, *args, **kwargs):
            index = module_self.index
            mask = DIM_FF_MASK[index]['mask']
            if mask is not None:
                x = x * mask
            out = module_self.forward_unprune(x, *args, **kwargs)
            return out
    module.forward = types.MethodType(prune_forward, module)


################################# utils for cross-granularity sparsity computaion #################################
def get_params_from_block(info_dict: Dict[int, Dict[str, List]]):
    r"""The calculation of sparsity.

    It calculates the mask and params of TransformerBlock, in a bottom-up way. This can support the cross-grained pruning.
    """
    res_exp, res_all = 0, 0
    for k in info_dict:
        modules = info_dict[k]['module']
        params = info_dict[k]['param']
        scores = info_dict[k]['score']

        # transformer_score * (att_score * (num_heads_mask * dim_head_mask * 4) + cross_att_score * (cross_num_heads_mask * cross_dim_head_mask * 4) + ffn_score * (dim_ff_score * param * 3))
        att_param_exp, cross_att_param_exp, ffn_param_exp = 0, 0, 0
        att_param_all, cross_att_param_all, ffn_param_all = 0, 0, 0
        if 'num_heads' in modules and 'dim_head' in modules:
            num_heads_index = modules.index('num_heads')
            num_heads_mask = scores[num_heads_index]
            dim_head_index = modules.index('dim_head')
            dim_head_mask = scores[dim_head_index]
            att_param_exp = torch.sum(num_heads_mask, dtype=torch.float) * torch.sum(dim_head_mask) * 4
            att_param_all = num_heads_mask.size(0) * dim_head_mask.size(0) * 4
        elif 'num_heads' in modules:
            num_heads_index = modules.index('num_heads')
            num_heads_mask = scores[num_heads_index]
            param = params[num_heads_index]
            att_param_exp = torch.sum(num_heads_mask, dtype=torch.float) * param * 4
            att_param_all = num_heads_mask.size(0) * param * 4
        elif 'dim_head' in modules:
            dim_head_index = modules.index('dim_head')
            dim_head_mask = scores[dim_head_index]
            param = params[dim_head_index]
            att_param_exp = torch.sum(dim_head_mask, dtype=torch.float) * param * 4
            att_param_all = dim_head_mask.size(0) * param * 4
        
        if 'cross_num_heads' in modules and 'cross_dim_head' in modules:
            cross_num_heads_index = modules.index('cross_num_heads')
            cross_num_heads_mask = scores[cross_num_heads_index]
            cross_dim_head_index = modules.index('cross_dim_head')
            cross_dim_head_mask = scores[cross_dim_head_index]
            cross_att_param_exp = torch.sum(cross_num_heads_mask, dtype=torch.float) * torch.sum(cross_dim_head_mask, dtype=torch.float) * 4
            cross_att_param_all = cross_num_heads_mask.size(0) * cross_dim_head_mask.size(0) * 4
        elif 'cross_num_heads' in modules:
            cross_num_heads_index = modules.index('cross_num_heads')
            cross_num_heads_mask = scores[cross_num_heads_index]
            param = params[cross_num_heads_index]
            cross_att_param_exp = torch.sum(cross_num_heads_mask, dtype=torch.float) * param * 4
            cross_att_param_all = cross_num_heads_mask.size(0) * param * 4
        elif 'cross_dim_head' in modules:
            cross_dim_head_index = modules.index('cross_dim_head')
            cross_dim_head_mask = scores[cross_dim_head_index]
            param = params[cross_dim_head_index]
            cross_att_param_exp = torch.sum(cross_dim_head_mask, dtype=torch.float) * param * 4
            cross_att_param_all = cross_dim_head_mask.size(0) * param * 4
                
        if 'dim_ff' in modules:
            dim_ff_index = modules.index('dim_ff')
            dim_ff_mask = scores[dim_ff_index]
            dim_ff_param = params[dim_ff_index]
            ffn_param_exp = torch.sum(dim_ff_mask, dtype=torch.float) * dim_ff_param * 3
            ffn_param_all = dim_ff_mask.size(0) * dim_ff_param * 3

        if 'att' in modules:
            att_index = modules.index('att')
            att_mask = scores[att_index]
            param = params[att_index]
            if att_param_exp == 0:
                att_param_exp = param * att_mask
            else:
                att_param_exp = (att_param_exp + param - att_param_all) * att_mask
            att_param_all = param
        
        if 'cross' in modules:
            cross_att_index = modules.index('cross_att')
            cross_att_mask = scores[cross_att_index]
            param = params[cross_att_index]
            if cross_att_param_exp == 0:
                cross_att_param_exp = param * cross_att_mask
            else:
                cross_att_param_exp = (cross_att_param_exp + param - cross_att_param_all) * att_mask
            cross_att_param_all = param
        
        if 'ffn' in modules:
            ffn_index = modules.index('ffn')
            ffn_mask = scores[ffn_index]
            param = params[ffn_index]
            if ffn_param_exp == 0:
                ffn_param_exp = param * ffn_mask
            else:
                ffn_param_exp = (ffn_param_exp + param - ffn_param_all) * att_mask
            ffn_param_all = param
        
        param_exp = att_param_exp + cross_att_param_exp + ffn_param_exp
        param_all = att_param_all + cross_att_param_all + ffn_param_all
        
        if 'transformer' in modules:
            transformer_index = modules.index('transformer')
            transformer_mask = scores[transformer_index]
            param = params[transformer_index]
            if param_exp == 0:
                param_exp = param * transformer_mask
                param_all = param
            else:
                param_exp = param_exp * transformer_mask

        res_exp += param_exp
        res_all += param_all
    return 1 - res_exp / res_all