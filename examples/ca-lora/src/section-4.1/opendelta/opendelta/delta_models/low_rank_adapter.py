
from opendelta.basemodel import DeltaBase
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.delta_models.layers.low_rank_linear import LowRankLinear
from opendelta.delta_models.layers.activations import Activations
from typing import Optional, Union
from opendelta.utils.signature import get_arg_names_inside_func
import torch.nn as nn
import torch
from typing import Optional
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import torch.nn as nn
import torch
import math
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)


class LowRankAdapterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~LowRankAdapterModel`

    """
    def __init__(
        self,
        reduction_factor=32,
        non_linearity="gelu_new",
        low_rank_w_init="glorot-uniform",
        low_rank_rank=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self,
                 reduction_factor=32,
                 non_linearity="gelu_new",
                 low_rank_w_init="glorot-uniform",
                 low_rank_rank=1,
                 device=None, 
                 backend='hf'):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.low_rank_w_init = low_rank_w_init
        self.low_rank_rank = low_rank_rank
        self.device = device
        self.instantiated = False
        self.backend=backend


    def instantiate(self, hiddens):
        self.hidden_dim = hiddens.shape[-1]
        self.hidden_dtype = hiddens.dtype

        self.down_sample_size = self.hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        self.down_sampler = LowRankLinear(self.hidden_dim, self.down_sample_size,
                                          w_init=self.low_rank_w_init,
                                          rank=self.low_rank_rank,
                                          dtype=self.hidden_dtype).to(self.device)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.hidden_dim,
                                        w_init=self.low_rank_w_init,
                                        rank=self.low_rank_rank,
                                        dtype=self.hidden_dtype).to(self.device)

        self.instantiated = True
        if self.backend == 'bmt':
            import bmtrain as bmt
            self.activation = bmt.BMTrainModelWrapper(self.activation)
            self.down_sampler = bmt.BMTrainModelWrapper(self.down_sampler)
            self.up_sampler = bmt.BMTrainModelWrapper(self.up_sampler)


    def post_forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the low-rank adapter,
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """

        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError

        if not self.instantiated:
            self.instantiate(hiddens = hiddens)

        z = self.down_sampler(hiddens)
        z = self.activation(z)
        adapter_output = self.up_sampler(z)

        modified_output = adapter_output + hiddens # residual_connection
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output






class LowRankAdapterModel(DeltaBase):
    r""" The implementation of LowRankAdapter, proposed as a baseline in
    `Compacter: Efficient Low-Rank Hypercomplex Adapter Layers <https://arxiv.org/abs/2106.04647>`_ .
    We found that it enjoys very few parameters but competitive performance, thus add it into OpenDelta.
    Low Rank Adapter parameterize each adapter’s weight as a product of two rank-one(low) weights.

    Add lowrank adapter layer to the designated ``modified_modules``. In sequential paradigm,  The modules' output is then
    passed into the low rank adapter's post_forward.

    .. note::
        We **assume** the output of the modified module is the hidden state or a tuple where hidden state is the
        first element. This is true for most PLMs. However, we admit that currently it's not rigorous, We will improve
        it in the next version. Currently, if you encount an error here for you backbone, you can modify the code to
        get the hidden state.

    All the hyperparameter is adopted from the `compacter code base <https://github.com/rabeehk/compacter>`_ .

    class attributes:
        - default_modified_modules = ["attn", "ff"] According to the compacter paper, we add low rank adapter to the attention layer
          and feed forward layer.
        - delta_type = "lowrankadapter"

    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified.
        reduction_factor (:obj:`int`, *optional*, default to ``16``): bottleneck_dim = hidden_dim//reduction_factor
        non_linearity (:obj:`str`, *optional*, default to ``"gelu_new"``): The non linearity activation used in between the down
                        projecter and the up projecter.
        low_rank_w_init (:obj:`str`, *optional*, default to ``"glorot-uniform"``): The weight init method of the factorized
                        linear weight.
        low_rank_rank (:obj:`int`, *optional*, default to 1): The rank of the low-rank decomposition.
        modified_modules (:obj:`List[str]`): For prefix tuning, the it must refer to an attention layer (Currently, only
                        the implemented ones)
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen
                         together with the prefix parameters.
        common_structure (:obj:`bool`, *optional*, default to :obj:`None`): whether using name-based addressing with a common structure mapping.

    """

    config_class = LowRankAdapterConfig
    delta_type = "low_rank_adapter"
    default_modified_modules = ["attn@.proj@", "ff@.w2@"]
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = True
    def __init__(self,
                 backbone_model: nn.Module,
                 reduction_factor = 32,
                 non_linearity = "gelu_new",
                 low_rank_w_init = "glorot-uniform",
                 low_rank_rank = 1,
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 backend: Optional[str] = 'hf',
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

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )


    # def add_all_delta_to_backbone(self,
    #                             module: nn.Module,
    #                             modified_modules: List[str],
    #                             ) -> nn.Module:
    #     for key, _ in module.named_modules():
    #         if self.find_key(key, modified_modules):
    #             self.update_module(module, key)
    #     self._pseudo_data_to_instantiate(module)
    #     self.mark_as_delta()
    #     return module

    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref, delta_module=adapterlayer, delta_name="low_rank_adapter")

    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = LowRankAdapter(reduction_factor = self.reduction_factor,
                                      non_linearity = self.non_linearity,
                                      low_rank_w_init = self.low_rank_w_init,
                                      low_rank_rank = self.low_rank_rank,
                                      device=module_device, backend=self.backend)
        self.delta_modules.append(adapterlayer)
        return adapterlayer
