from functools import partial
from typing import Optional, Union
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import torch.nn as nn
import torch
from opendelta.delta_models.layers.activations import Activations
import inspect
from opendelta.delta_models.layers.hypercomplex_linear import PHMLinear
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)

class HyperComplexAdapterLayer(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self,
                 reduction_factor=16,
                 non_linearity="relu",
                 phm_c_init="normal",
                 hypercomplex_division=4,
                 learn_phm=True,
                 hypercomplex_nonlinearity="glorot-uniform",
                 shared_phm_rule=False,
                 factorized_phm=True,
                 phm_rule: Optional[torch.Tensor]=None,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank=1,
                 phm_init_range=0.0001,
                 kronecker_prod=None,
                 device=None,
                 use_bias_up_sampler=True,
                 use_bias_down_sampler=True,
                 backend = 'hf',
                 ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.phm_c_init = phm_c_init
        self.hypercomplex_division = hypercomplex_division
        self.learn_phm = learn_phm
        self.phm_rule=phm_rule
        self.hypercomplex_nonlinearity = hypercomplex_nonlinearity
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm = factorized_phm
        self.shared_W_phm = shared_W_phm
        self.factorized_phm_rule = factorized_phm_rule
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod
        self.use_bias_up_sampler=use_bias_up_sampler
        self.use_bias_down_sampler=use_bias_down_sampler
        self.device = device
        self.backend = backend

        self.instantiated = False


    def instantiate(self, hiddens):
        self.hidden_dim = hiddens.shape[-1]
        self.hidden_dtype = hiddens.dtype
        self.down_sample_size = self.hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        self.down_sampler = PHMLinear(in_features=self.hidden_dim,
                                      out_features=self.down_sample_size,
                                      bias=self.use_bias_down_sampler,
                                      c_init=self.phm_c_init,
                                      phm_dim=self.hypercomplex_division,
                                      phm_rule=self.phm_rule,
                                      learn_phm=self.learn_phm,
                                      w_init=self.hypercomplex_nonlinearity,
                                      shared_phm_rule=self.shared_phm_rule,
                                      factorized_phm=self.factorized_phm,
                                      shared_W_phm=self.shared_W_phm,
                                      factorized_phm_rule=self.factorized_phm_rule,
                                      phm_rank=self.phm_rank,
                                      phm_init_range=self.phm_init_range,
                                      kronecker_prod=self.kronecker_prod,
                                      dtype = self.hidden_dtype).to(self.device)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.hidden_dim,
                                    bias=self.use_bias_up_sampler,
                                    c_init=self.phm_c_init,
                                    phm_dim=self.hypercomplex_division,
                                    phm_rule=self.phm_rule,
                                    learn_phm=self.learn_phm,
                                    w_init=self.hypercomplex_nonlinearity,
                                    shared_phm_rule=self.shared_phm_rule,
                                    factorized_phm=self.factorized_phm,
                                    shared_W_phm=self.shared_W_phm,
                                    factorized_phm_rule=self.factorized_phm_rule,
                                    phm_rank=self.phm_rank,
                                    phm_init_range=self.phm_init_range,
                                    kronecker_prod=self.kronecker_prod,
                                    dtype = self.hidden_dtype).to(self.device)
        self.instantiated = True
        if self.backend == "bmt":
            import bmtrain as bmt
            self.activation = bmt.BMTrainModelWrapper(self.activation)
            self.down_sampler = bmt.BMTrainModelWrapper(self.down_sampler)
            self.up_sampler = bmt.BMTrainModelWrapper(self.up_sampler)


    def post_forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the hypercomplex adapter,
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """

        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError

        if not self.instantiated:
            self.instantiate(hiddens=hiddens)


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

class CompacterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~CompacterModel`

    """
    def __init__(
        self,
        bottleneck_dim: Optional[int]=32,
        non_linearity: Optional[str]='relu',
        sequential: Optional[str] = True,
        reduction_factor=16,
        phm_c_init="normal",
        hypercomplex_division=4,
        learn_phm=True,
        hypercomplex_nonlinearity="glorot-uniform",
        shared_phm_rule=False,
        factorized_phm=True,
        shared_W_phm=False,
        factorized_phm_rule=False,
        phm_rank=1,
        phm_init_range=0.0001,
        kronecker_prod=None,
        use_bias_up_sampler=True,
        use_bias_down_sampler=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class CompacterModel(DeltaBase):
    r""" The implementation of `Compacter: Efficient Low-Rank Hypercomplex Adapter Layers <https://arxiv.org/abs/2106.04647>`_ .
    Add compacter layer to the designated ``modified_modules``. In sequential paradigm,  The modules' output is then
    passed into the compacter's post_forward.

    .. note::
        We **assume** the output of the modified module is the hidden state or a tuple where hidden state is the
        first element. This is true for most PLMs. However, we admit that currently it's not rigorous, We will improve
        it in the next version. Currently, if you encount an error here for you backbone, you can modify the code to
        get the hidden state.

    All the hyperparameter is adopted from the `compacter code base <https://github.com/rabeehk/compacter>`_ .

    class attributes:
        - default_modified_modules = ["attn", "ff"] According to the compacter paper, we add compacter to the attention layer
          and feed forward layer.
        - delta_type = "compacter"

    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified.
        modified_modules (:obj:`List[str]`): For prefix tuning, the it must refer to an attention layer (Currently, only
                        the implemented ones)
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen
                         together with the prefix parameters.
        common_structure (:obj:`bool`, *optional*, default to :obj:`None`): whether using name-based addressing with a common structure mapping.
        backend (:obj:`str`): choose the backend of plm, 'hf' for huggingface transformers,'bmt' for bmtrain
        reduction_factor (:obj:`int`, *optional*, default to ``16``): bottleneck_dim = hidden_dim//reduction_factor
        non_linearity (:obj:`str`, *optional*, default to ``"gelu_new"``): The non linearity activation used in between the down
                        projecter and the up projecter.
        phm_c_init (:obj:`str`, *optional*, default to ``"normal"``): The initialize method of the C in compacter.
        hypercomplex_division (:obj:`str`, *optional*, default to 4): The ``n`` in the paper. The number of division along a dimension in compector.
        learn_phm (:obj:`bool`, *optional*, default to :obj:`True` ): Whether the phm rule requires_grad. Note that we didn't check the performance of learn_phm=False.
        hypercomplex_nonlinearity (:obj:`str`, *optional*, default to ``"glorot-uniform"``): The initialize method of the W in compacter.
        shared_phm_rule (:obj:`str`, *optional* , default to :obj:`False`): Whether the phm rule is shared accross layer.
        factorized_phm (:obj:`str`, *optional*, default to :obj:`True`): Whether to factorize the phm into low rank product.
        shared_W_phm (:obj:`str`, *optional* , default to :obj:`False`): Whether the W_phm is shared accross layer.
        factorized_phm_rule (:obj:`str`, *optional* , default to :obj:`False`): Whether to factorize the phm rule into low rank product.
        phm_rank=1 (:obj:`int`, *optional*, default to 1): The rank of low rank decomposition of phm.
        phm_init_range (:obj:`float`, *optional*, default to 0.0001): The range of phm initialization.
        kronecker_prod (:obj:`bool`, *optional*, default to False): Whether to perform kronecker_prod in matvec_product, proposed by
            `Parameterization of Hypercomplex Multiplications <https://openreview.net/forum?id=rcQdycl0zyk>`_
        use_bias_up_sampler (:obj:`float`, *optional*, default to :obj:`True`): Whether add bias to the up projector.
                            Note that the bias for this is a ``hidden_dim`` vector.
        use_bias_down_sampler (:obj:`float`, *optional*, default to :obj:`True`): Whether add bias to the down projector.
                            Note that the bias for this is a ``bottleneck_dim`` vector.


    """
    config_class = CompacterConfig
    delta_type = "compacter"
    default_modified_modules = ["attn@.proj@", "ff@.w2@"]
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = True
    def __init__(self,
                 backbone_model,
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 backend: Optional[str] = 'hf',
                 reduction_factor=16,
                 non_linearity="gelu_new",
                 phm_c_init="normal",
                 hypercomplex_division=4,
                 learn_phm=True,
                 hypercomplex_nonlinearity="glorot-uniform",
                 shared_phm_rule=False,
                 factorized_phm=True,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank=1,
                 phm_init_range=0.0001,
                 kronecker_prod=None,
                 use_bias_up_sampler=True,
                 use_bias_down_sampler=True,
                ):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           exclude_modules=exclude_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           )
        assert shared_phm_rule == False, "In opendelta version {opendelta.__version__}, "\
            "shared_phm_rule is not supported. Later, sharing parameters will be tackled using"\
            "a unified paradigm."
        assert shared_W_phm == False, "In opendelta version {opendelta.__version__}, "\
            "shared_W_phm is not supported. Later, sharing parameters will be tackled using"\
            "a unified paradigm."
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )


    # def add_all_delta_to_backbone(self,
    #              module: nn.Module,
    #              modified_modules: List[str],
    #             ) -> nn.Module:
    #     for key, _ in module.named_modules():
    #         if self.find_key(key, modified_modules):
    #             self.update_module(module, key)
    #     self._pseudo_data_to_instantiate(module)
    #     self.mark_as_delta()
    #     return module

    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref,
                                      delta_module=adapterlayer,
                                      delta_name="compactor")

    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = HyperComplexAdapterLayer(reduction_factor=self.reduction_factor, non_linearity=self.non_linearity, phm_c_init=self.phm_c_init, hypercomplex_division=self.hypercomplex_division, learn_phm=self.learn_phm, hypercomplex_nonlinearity=self.hypercomplex_nonlinearity, shared_phm_rule=self.shared_phm_rule, factorized_phm=self.factorized_phm, shared_W_phm=self.shared_W_phm, factorized_phm_rule=self.factorized_phm_rule, phm_rank=self.phm_rank, phm_init_range=self.phm_init_range, kronecker_prod=self.kronecker_prod, use_bias_up_sampler=self.use_bias_up_sampler, use_bias_down_sampler=self.use_bias_down_sampler, device=module_device, backend=self.backend)
        self.delta_modules.append(adapterlayer)
        return adapterlayer
