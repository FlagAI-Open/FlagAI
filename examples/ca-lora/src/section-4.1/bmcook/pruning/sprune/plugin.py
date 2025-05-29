import torch
import bmtrain as bmt
from model_center.layer import TransformerBlock
from model_center.model import BaseModel

from .utils import set_pruning_att, set_pruning_ffn, set_pruning_transformer, get_params_from_block

class SPrunePlugin:
    r"""
    SPrunePlugin is a base class for structure prune in BMCook.
    All the modules supported by SprunePlugin includes: TransformerBlock layer, Attention layer, Feedforward layer, num_heads, dim_head, dim_ff.
    """
    def __init__(self, model: BaseModel):
        '''
        analyze the structure prune methed.
        '''
        # read some hyperparameters
        model_config = model.config
        dim_model = model_config.dim_model
        num_heads = model_config.num_heads
        dim_head = model_config.dim_head
        dim_ff = model_config.dim_ff
        if 'num_layers' in model_config.__dict__:
            num_layers = model_config.num_layers
            num_encoder_layers = num_layers
        elif 'num_encoder_layers' in model_config.__dict__:
            num_encoder_layers = model_config.num_encoder_layers
            num_decoder_layers = model_config.num_decoder_layers
        else:
            raise AttributeError("Missing num_layers or num_encoder_layers/num_decoder_layers in this config.")

        # model analysis
        prunable_all_params, self_att_num = 0, 0
        TRANSFORMER_MASK, FFN_MASK, ATT_MASK = [], [], []
        NUM_HEADS_MASK, DIM_HEAD_MASK, DIM_FF_MASK = [], [], []
        cross_att_buffer, cross_num_heads_buffer, cross_dim_head_buffer, names_buffer = [], [], [], []
        for name, module in model.named_modules():
            if type(module) in (bmt.block_layer.CheckpointBlock, TransformerBlock):
                block_type, overall_index = name.split('.')[0], int(name.split('.')[2])
                if block_type == 'decoder':
                    overall_index += num_encoder_layers

                ordered_parameters = [(k, v) for k, v in module.state_dict().items()]
                self_att_param, cross_att_param, ffn_param = 0, 0, 0
                for (k, v) in ordered_parameters:
                    if 'self_att' in k:
                        self_att_param += v.numel()
                    elif 'cross_att' in k:
                        cross_att_param += v.numel()
                    elif 'ffn' in k:
                        ffn_param += v.numel()
                
                # model ststistics
                transformer_layer_param = self_att_param + cross_att_param + ffn_param
                prunable_all_params += transformer_layer_param

                if transformer_layer_param > 0:
                    TRANSFORMER_MASK.append({
                                            'index': overall_index,
                                            'param': transformer_layer_param,
                                            'dim': 1,
                                            'mask': None
                                            })
                    set_pruning_transformer(module, len(TRANSFORMER_MASK)-1, TRANSFORMER_MASK, is_bmtCBlock=False)

                    if self_att_param > 0:
                        ATT_MASK.append({
                                                'index': overall_index, 
                                                'param': self_att_param,
                                                'dim': 1,
                                                'mask': None,
                                                })
                        NUM_HEADS_MASK.append({
                                                'index': overall_index, 
                                                'param': dim_head,
                                                'dim': num_heads,
                                                'mask': None
                                                })
                        DIM_HEAD_MASK.append({
                                                'index': overall_index, 
                                                'param': num_heads,
                                                'dim': dim_head,
                                                'mask': None
                                                })
                        set_pruning_att(module.self_att, len(ATT_MASK)-1, ATT_MASK, NUM_HEADS_MASK, DIM_HEAD_MASK)
                        self_att_num += 1
                                           
                    if cross_att_param > 0:
                        names_buffer.append(name)
                        cross_att_buffer.append({
                                                'index': overall_index, 
                                                'param': cross_att_param,
                                                'dim': 1,
                                                'mask': None,
                                                })
                        cross_num_heads_buffer.append({
                                                    'index': overall_index, 
                                                    'param': dim_head,
                                                    'dim': num_heads,
                                                    'mask': None
                                                    })
                        cross_dim_head_buffer.append({
                                                    'index': overall_index, 
                                                    'param': num_heads,
                                                    'dim': dim_head,
                                                    'mask': None
                                                    })

                    if ffn_param > 0:
                        FFN_MASK.append({
                                        'index': overall_index, 
                                        'param': ffn_param,
                                        'dim': 1,
                                        'mask': None,
                                        })
                        DIM_FF_MASK.append({
                                            'index': overall_index, 
                                            'param': dim_model,
                                            'dim': dim_ff,
                                            'mask': None
                                            })
                        set_pruning_ffn(module.ffn, len(FFN_MASK)-1, FFN_MASK, DIM_FF_MASK)

        # append cross_att to att and set pruning
        for (module_name, cross_att, cross_num_heads, cross_dim_head) in \
            zip(names_buffer, cross_att_buffer, cross_num_heads_buffer, cross_dim_head_buffer):

            ATT_MASK.append(cross_att)
            NUM_HEADS_MASK.append(cross_num_heads)
            DIM_HEAD_MASK.append(cross_dim_head)
            set_pruning_att(model.get_submodule(module_name).cross_att, len(ATT_MASK)-1, ATT_MASK, NUM_HEADS_MASK, DIM_HEAD_MASK)

        # check exception
        if TRANSFORMER_MASK == []:
            raise TypeError("plugin doesn't maintain any mask, all the mask lists are empty, \
                            please check if your model has the module type: bmt.CheckpointBlock or model_center.layer.TransformerBlock")
        elif any((FFN_MASK == [], ATT_MASK == [], NUM_HEADS_MASK ==[], DIM_HEAD_MASK == [], DIM_FF_MASK == [])):
            raise ValueError("Now BMCook doesn't support to prune model without feedforward layer or attention layer. It's also not allowed only layernorm parameters exist in these layers.")
        
        # init mask shape for the use of loga
        transformer_mask_shape = (len(TRANSFORMER_MASK))
        
        num_heads_list = [mask['dim'] for mask in NUM_HEADS_MASK]
        num_heads_layers, max_num_heads = len(NUM_HEADS_MASK), max(num_heads_list)
        num_heads_shape = (num_heads_layers, max_num_heads)
        num_heads_shape_mask = torch.stack(
            [(torch.arange(max_num_heads) < att['dim']).long() \
                for att in NUM_HEADS_MASK]
            )
        
        dim_head_list = [mask['dim'] for mask in DIM_HEAD_MASK]
        dim_head_layers, max_dim_head = len(DIM_HEAD_MASK), max(dim_head_list)
        dim_head_shape = (dim_head_layers, max_dim_head)
        dim_head_shape_mask = torch.stack(
            [(torch.arange(max_dim_head) < att['dim']).long() \
                for att in DIM_HEAD_MASK]
        )
        
        ffn_list = [mask['dim'] for mask in DIM_FF_MASK]
        ffn_layers, max_dim_ff = len(DIM_FF_MASK), max(ffn_list)
        ffn_shape = (ffn_layers, max_dim_ff)
        ffn_shape_mask = torch.stack(
            [(torch.arange(max_dim_ff) < ffn['dim']).long() \
                for ffn in FFN_MASK]
        )

        self.info_to_engine = {
            'all_params': prunable_all_params,
            'self_att_num': self_att_num,
            'num_encoder_layers': num_encoder_layers,
            'shape':{
                'transformer': ((transformer_mask_shape), torch.ones(transformer_mask_shape)),
                'att': ((num_heads_layers), torch.ones(num_heads_layers)),
                'ffn': ((ffn_layers), torch.ones(ffn_layers)),
                'num_heads': (num_heads_shape, num_heads_shape_mask),
                'dim_head': (dim_head_shape, dim_head_shape_mask),
                'dim_ff': (ffn_shape, ffn_shape_mask)
            }
        }

        self.transformer = TRANSFORMER_MASK
        self.att = ATT_MASK
        self.ffn = FFN_MASK
        self.num_heads = NUM_HEADS_MASK
        self.dim_head = DIM_HEAD_MASK
        self.dim_ff = DIM_FF_MASK

        del num_heads_list, dim_head_list, ffn_list, names_buffer, cross_att_buffer, cross_num_heads_buffer, cross_dim_head_buffer

    def print_masks(self, key: str = None):
        r"""print the masks managed in SPrunePlugin"""
        res = {'transformer': self.transformer,
                'att': self.att,
                'ffn': self.ffn,
                'num_heads': self.num_heads,
                'dim_head': self.dim_head,
                'dim_ff': self.dim_ff}
        if key is None:
            print(res)
        else:
            print(res[key])

    def save_plugin(self, path):
        r"""save the plugin as a dict.
        Args:
            path: `(str)`, the save path.
        """
        res = {'transformer': self.transformer,
                'att': self.att,
                'ffn': self.ffn,
                'num_heads': self.num_heads,
                'dim_head': self.dim_head,
                'dim_ff': self.dim_ff,
                'info_to_engine': self.info_to_engine}
        torch.save(res, path)

    def load_plugin(self, path):
        r"""load the saved dict to this plugin.
        Args:
            path: `(str)`, the file path.
        """
        plugin = torch.load(path)
        for k, v in plugin.items():
            self.__dict__[k] = v

    def training_masks(self):
        r"""
        traverse all the necessary masks in this training.
        """
        for name in self.info_to_engine['shape'].keys():
            for i, v in enumerate(self.__dict__[name]):
                if v['mask'] is not None:
                    yield name + '.' + str(i)
    
    def training_modules(self):
        r"""
        traverse all the necessary modules in this training.
        """
        for name in self.info_to_engine['shape'].keys():
            for _, v in enumerate(self.__dict__[name]):
                if v['mask'] is not None:
                    yield name
                    break

    def get_sparsity(self):
        r"""
        calculate the sparsity in single grain
        """
        info_list = {}

        for name in self.training_masks():
            module, index = name.split('.')[0], int(name.split('.')[1])
            param = self.__dict__[module][index]['param']
            mask = self.__dict__[module][index]['mask']
            index = self.__dict__[module][index]['index']
            if index not in info_list:
                info_list[index] = {'module': [module], 'param': [param], 'score': [mask]}
            else:
                if module in info_list[index]['module']:
                    module = 'cross_' + module
                info_list[index]['module'].append(module)
                info_list[index]['param'].append(param)
                info_list[index]['score'].append(mask)
        
        expected_params, all_params = get_params_from_block(info_list)

        sparsity = 1 - expected_params / all_params
        return sparsity