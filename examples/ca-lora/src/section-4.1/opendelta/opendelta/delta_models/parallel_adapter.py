from functools import partial
from random import random
from typing import Optional, Union
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import torch.nn as nn
import torch
from opendelta.delta_models.layers.activations import Activations
from opendelta import BaseDeltaConfig
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)

class ParallelAdapterLayer(nn.Module):
    r"""A layer of adapter tuning module. 
    """
    layer_count = 0

    @classmethod
    def count_layer(cls):
        cls.layer_count += 1
    
    @classmethod
    def get_layer_count(cls):
        return cls.layer_count

    def __init__(self, bottleneck_dim=24, non_linearity='gelu_new', scaled=1, device=None, backend='hf'):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self.instantiated = False
        self.non_linearity = non_linearity
        self.scaled = scaled
        self.backend = backend
        
        self.layer_id = ParallelAdapterLayer.get_layer_count()
        ParallelAdapterLayer.count_layer()
        
    
    def instantiate(self, hiddens):
        self.hidden_dim =  hiddens.shape[-1]
        self.hidden_dtype = hiddens.dtype
        self.modulelist = nn.Sequential()
        self.modulelist.add_module("down_proj",nn.Linear(self.hidden_dim, self.bottleneck_dim, device=self.device, dtype=self.hidden_dtype))

        # select non-linearity
        self.modulelist.add_module("non_linear", Activations(self.non_linearity.lower()))

        self.modulelist.add_module("up_proj", nn.Linear(self.bottleneck_dim, self.hidden_dim,  device=self.device, dtype=self.hidden_dtype))

        self.instantiated = True
        # initialize the weight, which is important for fast convergence and better performance. 
        self.apply(self._init_weight)
        if self.backend == 'bmt':
            import bmtrain as bmt
            self.modulelist = bmt.BMTrainModelWrapper(self.modulelist)
    
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01) 
            if module.bias is not None:
                module.bias.data.zero_()
        
    
    def pre_forward(self, *args, **kwargs):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter, 
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """
        if isinstance(args, tuple):
            hiddens = args[0]
        elif isinstance(args, torch.Tensor):
            hiddens = args
        else:
            raise TypeError


        if not self.instantiated:
            # logger.debug(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hiddens = hiddens)
                

        self.adapter_output = self.modulelist(hiddens) * self.scaled
        return args, kwargs

    def post_forward(self, output, **kwargs):
        if isinstance(output, tuple):
            hidden = output[0]
        elif isinstance(output, torch.Tensor):
            hidden = output
        else:
            raise TypeError

        modified_output = self.adapter_output + hidden
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output
    
  

class ParallelAdapterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~ParallelAdapterModel`

    """
    def __init__(
        self, 
        bottleneck_dim: Optional[int]=24, 
        non_linearity: Optional[str]='gelu_new',
        scaled: Optional[float]=1.,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class ParallelAdapterModel(DeltaBase):
    r""" The implementation of Parallel Adapter(`TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING <https://arxiv.org/abs/2110.04366>`_ ) .
    Add adapter to the designated ``modified_modules``. In parallel paradigm, The modules' output is then passed into the adapter's 
    post_forward. 
    
    .. note::
        We **assume** the output of the modified module is the hidden state or a tuple where hidden state is the 
        first element. This is true for most PLMs. However, we admit that currently it's not rigorous, We will improve 
        it in the next version. Currently, if you encount an error here for you backbone, you can modify the code to 
        get the hidden state.

    class attributes:
        - default_modified_modules = ["attn", "ff"] According to the Adapter paper, we add adapter to the attention layer
          and feed forward layer. 
        - delta_type = "adapter"

    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified. 
        bottleneck_dim (:obj:`int`): The dimension of the adapter's bottleneck. 
        non_linearity (:obj:`str`): The non linearity of the adapter.
        modified_modules (:obj:`List[str]`): modules to add parallel adapter. Must be paired and have the save order in layer. For examples, ["attn", "attn", "ff.w1", "ff.w2"] add one parallel adapter from attn's input to attn's output, and another one from ff.w1's input to ff.w2's output.
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen together with the parallel adapter parameters.
        common_structure (:obj:`bool`): whether using name-based addressing witha common structure mapping.
        backend (:obj:`str`): choose the backend of plm, 'hf' for huggingface transformers,'bmt' for bmtrain

    """
    config_class = ParallelAdapterConfig
    delta_type = "parallel_adapter"
    default_modified_modules = ["attn@", "attn@", "ff@.w1@", "ff@.w2@"]
    # default_modified_modules = ["attn", "attn", "ff.w1", "ff.w2"]
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = True
    def __init__(self,
                 backbone_model: nn.Module, 
                 bottleneck_dim: Optional[int]=24, 
                 non_linearity: Optional[str]='gelu_new',
                 modified_modules: Optional[bool] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[bool] = None,
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

        self.delta_modules = nn.ModuleList()

        self.ith = 0
        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )
  
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        if self.ith % 2 == 0:
            adapterlayer = self.new_module_like(ref)
            self.insert_module(ref, "before", delta_module=adapterlayer, delta_name="parallel_adapter")
        if self.ith % 2 == 1 or self.modified_modules[self.ith] == self.modified_modules[self.ith + 1]:
            adapterlayer = self.delta_modules[-1]
            self.insert_module(ref, "after", delta_module=adapterlayer, delta_name="parallel_adapter")
            self.ith |= 1
        self.ith += 1
        self.ith %= len(self.modified_modules)
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = ParallelAdapterLayer(bottleneck_dim=self.bottleneck_dim, non_linearity=self.non_linearity, device=module_device, backend=self.backend)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    