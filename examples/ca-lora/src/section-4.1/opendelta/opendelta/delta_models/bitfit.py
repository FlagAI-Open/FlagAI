from typing import Optional, Union
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase, is_leaf_module
from opendelta.utils.cuda import get_device, get_dtype
import torch.nn as nn

import torch
from torch.nn import init
import math
from opendelta import BaseDeltaConfig
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)


class BitFitConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~BitFitModel`

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])

class BiasLayer(nn.Module):
    def __init__(self, init_method="zero", dtype=None, device=None, backend=None):
        super().__init__()
        self.init_method=init_method
        self.instantiated = False
        self.dtype = dtype
        self.device = device
        self.backend = backend

    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.bias = nn.Parameter(torch.zeros(hidden_dim, dtype=self.dtype, device=self.device))
        else:
            raise NotImplementedError
        self.instantiated = True
        if self.backend == 'bmt':
            import bmtrain as bmt
            self.bias = bmt.BMTrainModelWrapper(self.bias)

    def post_forward(self, output):
        r"""Presuming the first argument is the tensor to add bias along the last dimension.
        In most cases, it is correct. However, be aware of the possibility that the presumption
        doesn't hold.
        """
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError

        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            logger.debug(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)

        modified_output = hiddens + self.bias

        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output



class BitFitModel(DeltaBase):
    r""" The implementation of `BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models <https://arxiv.org/abs/2106.10199>`_ .
    Unfreeze bias term (or add bias term if bias term is absent in the backbone, e.g. T5) to the modules of
    a transformer block.

    .. note::

        **Broadcast to Submodule**: We modify all potential positions  of the specified
        ``modified_modules``. That is to say, if we specify ``attn`` in the modified_modules, then all position
        including the q, k, v and out linear layer of the attention layer are added bias layer (or unfreezing).
        The potential position is determined according to equation (1)-(5) and the previous three
        equations.


    class attributes:
        - default_modified_modules = ["attn", "ff", "layer_norm","lm_head.proj"] According to the paper and the
          implementation in `Compacter's baseline <https://github.com/rabeehk/compacter>`_ , we modify the
          bias term in the above modules.
        - delta_type = "bitfit"




    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified.
        modified_modules (:obj:`List[str]`): For prefix tuning, the it must refer to an attention layer (Currently, only
                        the implemented ones)
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen
                         together with the prefix parameters.
        common_structure (:obj:`bool`): whether using name-based addressing with a common structure mapping.

    """


    config_class = BitFitConfig
    delta_type = "bitfit"
    default_modified_modules = ["attn@", "ff@", "layer_norm@","lm_head@.proj@"] # modify all the bias parameter in attention and feed-forward layer.
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = False
    def __init__(self,
                 backbone_model: nn.Module,
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 backend: Optional[str] = "hf",
                 ):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           exclude_modules=exclude_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           backend=backend,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_params = nn.ParameterList()
        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                       self.modified_modules)
        
        


    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        self.modify_module(ref)


    def modify_module(self,
                      module: nn.Module,
                      ):
        if is_leaf_module(module):
            if self.backend_mapping.check_type(module, 'linear') or \
                self.backend_mapping.check_type(module, 'layer_norm'):
                self.add_bias_to_modules_have_bias_or_known_type(module)
            else:
                self.add_bias_to_others(module)
        else:
            for n, c in module.named_modules():
                self.add_bias_to_modules_have_bias_or_known_type(c)

    def add_bias_to_modules_have_bias_or_known_type(self, c):
        '''If it has bias, unfreeze it. 
        If it doesn't have bias: if it is Linear of LN, add to it, else pass.
        '''
        if 'bias' in [n for n,p in c.named_parameters()]:
            c.bias.requires_grad = True
            self.delta_params.append(c.bias)
        else:
            if self.backend_mapping.check_type(c, 'linear') or \
                self.backend_mapping.check_type(c, 'layer_norm'): 
                bias = nn.Parameter(torch.empty(c.out_features), requires_grad=True)
                
                self._reset_bias_parameters(c, bias) 
                if self.backend == 'bmt':
                    import bmtrain as bmt
                    bias = bmt.BMTrainModelWrapper(bias)
            
                c.register_parameter('bias', bias)
                self.delta_params.append(bias)

    def add_bias_to_others(self, c): 
        new_bias = BiasLayer(dtype=get_dtype(c), device=get_device(c), backend=self.backend)

        self.insert_sequential_module(c, delta_module=new_bias, delta_name="bitfit") # name shouldn't be `bias` here, since the name `bias` is reserved for some module such as roberta's LayerNorm.
        self.delta_modules.append(new_bias)

    @staticmethod
    def _reset_bias_parameters(linear_module, bias):
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(bias, -bound, bound)
        # init.uniform_(bias, -bound, bound)

    def detach(self, module):
        r"""Not implemented for BitFit yet. Please wait for the next version.
        """
        raise NotImplementedError

    def attach(self, module):
        r"""Not implemented for BitFit yet. Please wait for the next version.
        """
        raise NotImplementedError
