import torch
import bmtrain as bmt

from collections import OrderedDict

def _save_to_state_dict(model : torch.nn.Module, destination, prefix):
    if isinstance(model, bmt.CheckpointBlock):
        if bmt.global_var.config['rank'] != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model.state_dict(destination, prefix, False)
    else:
        if bmt.global_var.config['rank'] != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model._save_to_state_dict(destination, prefix, False)

def _save_to_rank0(model : torch.nn.Module, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=model._version)
    _save_to_state_dict(model, destination, prefix)
    for name, module in model._modules.items():
        if module is not None:
            _save_to_rank0(module, destination, prefix + name + '.')
    for hook in model._state_dict_hooks.values():
        hook_result = hook(model, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination

def save_spruned(model, plugin, file_name):
    # get state_dict of model
    torch.cuda.synchronize()
    state_dict = _save_to_rank0(model)

    new_state_dict = state_dict.copy()
    config = model.config
    
    # reformat masks
    num_encoder = plugin.info_to_engine['num_encoder_layers']
    att_boundary = plugin.info_to_engine['att_boundary']
    prune_layer = []
    num_heads_mask, dim_head_mask, dim_ff_mask = {}, {}, {}
    for mask in plugin.transformer:
        if mask['mask'] == 0:
            layer_index = mask['index']
            if layer_index < num_encoder:
                prune_layer.append('encoder.layers.' + str(layer_index))
            else:
                prune_layer.append('decoder.layers.' + str(layer_index - num_encoder))
    
    for i, mask in enumerate(plugin.att):
        layer_index = mask['index']
        if layer_index < num_encoder:
            layer_prefix = 'encoder.layers.' + str(layer_index) + '.self_att'
        else:
            if i < att_boundary:
                layer_prefix = 'decoder.layers.' + str(layer_index - num_encoder) + '.self_att'
            else:
                layer_prefix = 'decoder.layers.' + str(layer_index - num_encoder) + '.cross_att'
        if mask['mask'] == 0:       # the att_layer pruned
            prune_layer.append(layer_prefix)
        else:
            if plugin.num_heads[i]['mask'] is not None:
                num_heads_mask[layer_prefix + '.self_attention'] = plugin.num_heads[i]['mask']
            if plugin.dim_head[i]['mask'] is not None:
                dim_head_mask[layer_prefix + '.self_attention'] = plugin.dim_head[i]['mask']
    
    for j, mask in enumerate(plugin.ffn):
        layer_index = mask['index']
        if layer_index < num_encoder:
            layer_prefix = 'encoder.layers.' + str(layer_index) + '.ffn'
        else:
            layer_prefix = 'decoder.layers.' + str(layer_index - num_encoder) + '.ffn'
        if mask['mask'] == 0:
            prune_layer.append(layer_prefix)
        elif plugin.dim_ff[j]['mask'] is not None:
            dim_ff_mask[layer_prefix + '.ffn'] = plugin.dim_ff[j]['mask']

    # do pruning
    find_first_head = None
    for k, v in state_dict.items():
        if any([k.startswith(prefix) for prefix in prune_layer]):
            del new_state_dict[k]
    
        elif 'encoder.layers' in k or 'decoder.layers' in k:
            if 'project_' in k and k.split('.project_')[0] in num_heads_mask:              
                k_prefix = k.split('.project_')[0]

                if find_first_head is None:
                    find_first_head  = num_heads_mask[k_prefix]
                
                heads_mask = num_heads_mask[k_prefix]['mask']
                dim_head = num_heads_mask[k_prefix]['param']
                num_heads_unprune = num_heads_mask[k_prefix]['dim']
                num_heads_target = torch.sum(heads_mask).item()
                v = v.view(num_heads_unprune, dim_head, config.dim_model)
                tgt = []
                for i, mask in enumerate(heads_mask):
                    if mask == 1.:
                        tgt.append(v[i])
                v = torch.stack(tgt)
                assert v.size() == (num_heads_target, dim_head, config.dim_model)
                new_state_dict[k] = v.view(num_heads_target * dim_head, config.dim_model)
            
            elif 'attention_out' in k and k.split('.attention_out')[0] in num_heads_mask:
                k_prefix = k.split('.attention_out')[0]

                param_info = num_heads_mask[k_prefix]
                heads_mask = param_info['mask']
                dim_head = param_info['param']
                num_heads_unprune = param_info['dim']
                num_heads_target = torch.sum(heads_mask).item()
                v = v.permute(1, 0).view(num_heads_unprune, dim_head, config.dim_model)
                tgt = []
                for i, mask in enumerate(heads_mask):
                    if mask == 1.:
                        tgt.append(v[i])
                v = torch.stack(tgt)
                assert v.size() == (num_heads_target, dim_head, config.dim_model)
                new_state_dict[k] = v.view(num_heads_target * dim_head, config.dim_model).permute(1, 0)
            
            elif 'ffn.ffn.w_in' in k and k.split('.ffn.ffn.w_in')[0] in dim_ff_mask:
                k_prefix = k.split('.ffn.ffn.w_in')[0]
                
                dimff_mask = dim_ff_mask[k_prefix]['mask']
                dimff_unprune = dim_ff_mask[k_prefix]['dim']
                dimff_target = torch.sum(dimff_mask).item()
                assert v.size() == (dimff_unprune, config.dim_model)
                tgt = []
                for i, mask in enumerate(dimff_mask):
                    if mask == 1.:
                        tgt.append(v[i])
                v = torch.stack(tgt)
                assert v.size() == (dimff_target, config.dim_model)
                new_state_dict[k] = v
            
            elif 'ffn.ffn.w_out' in k and k.split('.ffn.ffn.w_out')[0] in dim_ff_mask:
                k_prefix = k.split('.ffn.ffn.w_out')[0]
                
                dimff_mask = dim_ff_mask[k_prefix]['mask']
                dimff_unprune = dim_ff_mask[k_prefix]['dim']
                dimff_target = torch.sum(dimff_mask).item()
                v = v.permute(1, 0)
                assert v.size() == (dimff_unprune, config.dim_model)
                tgt = []
                for i, mask in enumerate(dimff_mask):
                    if mask == 1.:
                        tgt.append(v[i])
                v = torch.stack(tgt).permute(1, 0)
                assert v.size() == (config.dim_model, dimff_target)
                new_state_dict[k] = v
        
        if 'position_bias.relative_attention_bias' in k:
            num_heads_unprune = find_first_head['dim']
            heads_mask = find_first_head['mask']
            v = v.permute(1, 0).view(num_heads_unprune, -1)
            tgt = []
            for i, mask in enumerate(heads_mask):
                if mask == 1.:
                    tgt.append(v[i])
            v = torch.stack(tgt)
            new_state_dict[k] = v.permute(1, 0)

    if bmt.global_var.config["rank"] == 0:
        torch.save(new_state_dict, file_name)