from functools import partial
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.signature import get_arg_names_inside_func, signature
from typing import Optional, Union
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.roberta.modeling_roberta import RobertaAttention
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
from transformers.models.t5 import T5ForConditionalGeneration
import torch.nn as nn
import torch
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)

# We are going to refactor the code of Prefix Tuning.

class PrefixLayerT5(nn.Module):
    r"""A layer of prefix tuning module. The layer's forward function pass (or concatenate) the additional past_key_value
    into the original attention layer's forward function.
    """
    def __init__(self, prefix_token_num, num_heads, device,):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.num_heads = num_heads
        self.device = device
        self.instantiated = False

    def instantiate(self, hidden_dim):
        self.past_key = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_value = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_key_reparam = None
        self.past_value_reparam = None
        self.instantiated = True


    def pre_forward(self, *args, **kwargs):
        r"""The args and kwargs are inherited from the T5Attention's forward function.
        """
        batch_size = args[0].shape[0]
        seq_len = args[0].shape[-2]
        if not self.instantiated:
            self.hidden_dim = args[0].shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
        if self.past_key_reparam is None:
            past_key = self.past_key
        else:
            past_key = self.past_key_reparam
        if self.past_value_reparam is None:
            past_value = self.past_value
        else:
            past_value = self.past_value_reparam


        def expand_batchsize(x):
            x = x.reshape(self.prefix_token_num, self.num_heads, -1).transpose(0,1)
            x = x.unsqueeze(0).expand(batch_size, *x.shape)
            return x

        if 'position_bias' in kwargs and kwargs['position_bias'] is not None:
            if kwargs['position_bias'].shape[-1] != seq_len + self.prefix_token_num: # Then the position_bias should be re-calculated
                kwargs['position_bias'] = None
        if kwargs['past_key_value'] is None:
            kwargs['past_key_value'] = (expand_batchsize(past_key), expand_batchsize(past_value))

        past_key_len = kwargs['past_key_value'][0].shape[-2]

        if 'mask' in kwargs and kwargs['mask'] is not None:
            mask_len = kwargs['mask'].shape[-1]
            if past_key_len + seq_len == mask_len + self.prefix_token_num:

                am = kwargs['mask']  # Should check the format of the attention_mask when moving to a new plm.
                kwargs['mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
        return args, kwargs

    def post_forward(self, output):
        r""" Remove the cached positional bias, since the next layer may not have prefix token.
        """
        output = output[:2] + (None, )+ output[3:]
        return output

class PrefixLayerBart(nn.Module):
    r"""A layer of prefix tuning module. The layer's forward function pass (or concatenate) the additional past_key_value
    into the original attention layer's forward function.
    """
    def __init__(self, prefix_token_num, num_heads, device,):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.num_heads = num_heads
        self.device = device
        self.instantiated = False

    def instantiate(self, hidden_dim):
        self.past_key = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_value = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_key_reparam = None
        self.past_value_reparam = None
        self.instantiated = True


    def pre_forward(self, *args, **kwargs):
        r"""The args and kwargs are inherited from the T5Attention's forward function.
        """

        batch_size = kwargs['hidden_states'].shape[0]
        if not self.instantiated:
            self.hidden_dim = kwargs['hidden_states'].shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
        if self.past_key_reparam is None:
            past_key = self.past_key
        else:
            past_key = self.past_key_reparam
        if self.past_value_reparam is None:
            past_value = self.past_value
        else:
            past_value = self.past_value_reparam

        # from IPython import embed
        # embed()
        def expand_batchsize(x):
            x = x.reshape(self.prefix_token_num, self.num_heads, -1).transpose(0,1)
            x = x.unsqueeze(0).expand(batch_size, *x.shape)
            return x
        # from IPython import embe

        if 'past_key_value' not in kwargs or kwargs['past_key_value'] is None:
            kwargs['past_key_value'] = (expand_batchsize(past_key), expand_batchsize(past_value))

        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
        return args, kwargs


class PrefixLayerGPT2(nn.Module):
    r"""A layer of prefix tuning module. The layer's forward function pass (or concatenate) the additional past_key_value
    into the original attention layer's forward function.
    """
    def __init__(self, prefix_token_num, num_heads, device,):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.num_heads = num_heads
        self.device = device
        self.instantiated = False

    def instantiate(self, hidden_dim):
        self.past_key = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_value = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_key_reparam = None
        self.past_value_reparam = None
        self.instantiated = True


    def pre_forward(self, *args, **kwargs):
        r"""The args and kwargs are inherited from the T5Attention's forward function.
        """
        batch_size = args[0].shape[0]
        if not self.instantiated:
            self.hidden_dim = args[0].shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
        if self.past_key_reparam is None:
            past_key = self.past_key
        else:
            past_key = self.past_key_reparam
        if self.past_value_reparam is None:
            past_value = self.past_value
        else:
            past_value = self.past_value_reparam

        def expand_batchsize(x):
            x = x.reshape(self.prefix_token_num, self.num_heads, -1).transpose(0,1)
            x = x.unsqueeze(0).expand(batch_size, *x.shape)
            return x


        if kwargs['layer_past'] is None:
            kwargs['layer_past'] = (expand_batchsize(past_key), expand_batchsize(past_value))
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
        return args, kwargs



class PrefixLayerDistilBert(nn.Module):
    # TODO: Warning: have bugs
    def __init__(self, prefix_token_num, device,):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.device = device
        self.key_instantiated = False
        self.value_instantiated = False

    def forward(self, *args, **kwargs):
        mask = kwargs["mask"]
        key, value = kwargs['key'], kwargs['value']
        prefix_mask = torch.ones(mask.shape[0], self.prefix_token_num, dtype=mask.dtype, device=mask.device)
        concated_mask = torch.cat([prefix_mask, mask], dim=1)
        pseudo_prefix = torch.zeros(key.shape[0], self.prefix_token_num, key.shape[2], dtype=key.dtype, device=key.device)
        concated_key = torch.cat([pseudo_prefix, key], dim=1)
        concated_value = torch.cat([pseudo_prefix, value], dim=1)
        kwargs["mask"] = concated_mask
        kwargs['key'] = concated_key
        kwargs['value'] = concated_value
        return args, kwargs


    def key_instantiate(self, hidden_dim):
        self.past_key = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_key_reparam = None
        self.key_instantiated = True

    def value_instantiate(self, hidden_dim):
        self.past_value = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_value_reparam = None
        self.value_instantiated = True

    def key_pre_forward(self, *args, **kwargs):
        _input = args[0]
        _input = _input[:,self.prefix_token_num:, :]
        args = (_input,) +args[1:]
        return args, kwargs

    def value_pre_forward(self, *args, **kwargs):
        _input = args[0]
        _input = _input[:,self.prefix_token_num:, :]
        args = (_input,) +args[1:]
        return args, kwargs

    def key_forward(self, output: torch.Tensor):  ### Check whether run prefix is ok, 12.21
        if isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        if not self.key_instantiated:
            self.hidden_dim = hiddens.shape[-1]
            logger.debug(f"Got key hidden dim hidden_dim {self.hidden_dim}")
            self.key_instantiate(hidden_dim=self.hidden_dim)
        batch_size = hiddens.shape[0]
        if self.past_key_reparam is None:
            past_key = self.past_key
        else:
            past_key = self.past_key_reparam
        output = torch.cat([past_key.unsqueeze(0).expand(batch_size, *past_key.shape), hiddens], dim=1)
        return output

    def value_forward(self, output: torch.Tensor):  ### Check whether run prefix is ok, 12.21
        if isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        if not self.value_instantiated:
            self.hidden_dim = hiddens.shape[-1]
            logger.debug(f"Got value hidden dim hidden_dim {self.hidden_dim}")
            self.value_instantiate(hidden_dim=self.hidden_dim)
        batch_size = hiddens.shape[0]
        if self.past_value_reparam is None:
            past_value = self.past_value
        else:
            past_value = self.past_value_reparam
        output = torch.cat([past_value.unsqueeze(0).expand(batch_size, *past_value.shape), hiddens], dim=1)
        return output


class PrefixLayerBert(nn.Module):
    r"""A layer of prefix tuning module. The layer's forward function pass (or concatenate) the additional past_key_value
    into the original attention layer's forward function.
    """
    def __init__(self, prefix_token_num, num_heads, device,):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.num_heads = num_heads
        self.device = device
        self.instantiated = False

    def instantiate(self, hidden_dim):
        self.past_key = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_value = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_key_reparam = None
        self.past_value_reparam = None
        self.instantiated = True


    def pre_forward(self, *args, **kwargs):
        r"""The args and kwargs are inherited from the T5Attention's forward function.
        """
        batch_size = args[0].shape[0]
        if not self.instantiated:
            self.hidden_dim = args[0].shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
        if self.past_key_reparam is None:
            past_key = self.past_key
        else:
            past_key = self.past_key_reparam
        if self.past_value_reparam is None:
            past_value = self.past_value
        else:
            past_value = self.past_value_reparam


        def expand_batchsize(x):
            x = x.reshape(self.prefix_token_num, self.num_heads, -1).transpose(0,1)
            x = x.unsqueeze(0).expand(batch_size, *x.shape)
            return x
        # from IPython import embe

        if 'past_key_value' not in kwargs or kwargs['past_key_value'] is None:
            kwargs['past_key_value'] = (expand_batchsize(past_key), expand_batchsize(past_value))

        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
        elif len(args) >1: # attention mask is passed via positional argument
            am = args[1]
            am = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
            args = (args[0], am) + args[2:]
        # from IPython import embed
        # embed(header = "Herein prefixroberta")
        return args, kwargs



class PrefixLayerRoberta(nn.Module):
    r"""A layer of prefix tuning module. The layer's forward function pass (or concatenate) the additional past_key_value
    into the original attention layer's forward function.
    """
    def __init__(self, prefix_token_num, num_heads, device,):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.num_heads = num_heads
        self.device = device
        self.instantiated = False

    def instantiate(self, hidden_dim):
        self.past_key = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_value = nn.Parameter(torch.randn(self.prefix_token_num, hidden_dim, device=self.device), requires_grad=True)
        self.past_key_reparam = None
        self.past_value_reparam = None
        self.instantiated = True


    def pre_forward(self, *args, **kwargs):
        r"""The args and kwargs are inherited from the T5Attention's forward function.
        """
        batch_size = args[0].shape[0]
        if not self.instantiated:
            self.hidden_dim = args[0].shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
        if self.past_key_reparam is None:
            past_key = self.past_key
        else:
            past_key = self.past_key_reparam
        if self.past_value_reparam is None:
            past_value = self.past_value
        else:
            past_value = self.past_value_reparam

        # from IPython import embed
        # embed()
        def expand_batchsize(x):
            x = x.reshape(self.prefix_token_num, self.num_heads, -1).transpose(0,1)
            x = x.unsqueeze(0).expand(batch_size, *x.shape)
            return x
        # from IPython import embe

        if 'past_key_value' not in kwargs or kwargs['past_key_value'] is None:
            kwargs['past_key_value'] = (expand_batchsize(past_key), expand_batchsize(past_value))

        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
        elif len(args) >1: # attention mask is passed via positional argument
            am = args[1]
            am = torch.cat([-torch.zeros((*am.shape[:-1],self.prefix_token_num), dtype = am.dtype,device=am.device), am], dim=-1)
            args = (args[0], am) + args[2:]
        # from IPython import embed
        # embed(header = "Herein prefixroberta")
        return args, kwargs



    # def post_forward(self, output):
    #     r""" Remove the cached positional bias, since the next layer may not have prefix token.
    #     """
    #     output = output[:2] + (None, )+ output[3:]
    #     return output


class ReparameterizeFunction(nn.Module):
    r""" Prefix Tuning's performance is better with a reparameterize module, which generates
    the ``past_key_value`` using an MLP instead of directly optimizing the ``past_key_value`` as leaf variable.
    In our implementation, the reparameterize module is constructed according to the number of parameters
    in all ``past_key_value``s. Thus, variable number of prefixlayer is supported (not restricting to being equal
    to the number of layers of the pretraind language model)


    """
    def __init__(self, prefix_token_num, embed_dim,  dropout_rate=0.0, mid_dim=512, module_list=[]):
        super().__init__()
        self.prefix_token_num = prefix_token_num
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.module_list = module_list
        self.dropout = nn.Dropout(dropout_rate)
        self.record_parameters()
        self.compatibility_check()
        self.define_reparameterization_network()

    def record_parameters(self):
        r""" Enumerate the parameters that need to be reparameterized.
        Then, delete the original parameters.
        """
        tot = 0
        for module in self.module_list:
            for n, parameters in module.named_parameters():
                tot += parameters.numel()
                module.register_parameter(n, None)
        self.total_parameters_num = tot

    def compatibility_check(self,):
        r"""May be removed.
        """
        assert self.total_parameters_num % self.prefix_token_num == 0

    def allocate_parameter(self):
        r""" At the beginning of each forward pass through the whole network(PLM),
        cacalulate the reparameterized past_key and past_value (``past_key_reparam`` and ``past_value_reparam``)
        for later use in each layer.
        """
        input_tokens = self.input_tokens
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)
        seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(seqlen, len(self.module_list) * 2, self.module_list[0].hidden_dim)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([1, 0, 2]).split(2)

        for module_id, module in enumerate(self.module_list):
            module.past_key_reparam = past_key_values[module_id][0]
            module.past_value_reparam = past_key_values[module_id][1]

    def pre_forward(self, *args, **kwargs):
        r""" Firstly forward through the reparameterized network, and then go through normal forward pass of the PLM.
        """
        self.allocate_parameter()
        return args, kwargs

    def define_reparameterization_network(self) -> None:
        r""" Build the reparameterize module
        """
        self.input_tokens = nn.Parameter(torch.arange(self.prefix_token_num).long(), requires_grad=False) # to allow automatic devicing
        self.wte = nn.Embedding(self.prefix_token_num, self.embed_dim)
        self.control_trans = nn.Sequential(
            nn.Linear(self.embed_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.total_parameters_num//self.prefix_token_num)
        )


class PrefixConfig(BaseDeltaConfig):
    def __init__(
        self,
        prefix_token_num=6,
        reparameterize=True,
        embed_dim: Optional[int]=512,
        mid_dim: Optional[int]=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])





class PrefixModel(DeltaBase):
    r""" The implementation of `Prefix-Tuning: Optimizing Continuous Prompts for Generation <https://arxiv.org/abs/2101.00190>`_ .
    However, as attention block of different PLM differs substantially, e.g., the input arguments, the name convention
    of ``past_key_value``, we have to implement different prefixlayer for different PLM. Given the inconvenience in the
    code level, we only support several commonly used backbone models (Currently: T5, DistilBert,Bert, Roberta, GPT2,
    BART). If you are trying to apply delta tuning to other backbone models, we suggest you trying other delta models
    or implementing it and making a pull request.

    Experimental Feature:

        Support inserting prefix token before each layer. For example, layer 3 4 6 10 and other layer untouched.

    .. note::

        If using reparameterize, the parameters will be in a reparameterization network, not in the prefix, which
        we attach to the first prefix layer. We will add a function to save only the generated prefix parameters for
        saving in the next version.



    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified.
        prefix_token_num (:obj:`int`): the number of prefix token
        reparameterize (:obj:`bool`): Whether use the reparameterization for prefix tuning.
        embed_dim (:obj:`int`): The embeding dimension of prefix token when using the reparameterization.
        mid_dim (:obj:`int`): The dimension of the hiddens of the reparameterization network.
        modified_modules (:obj:`List[str]`): For prefix tuning, the it must refer to an attention layer (Currently, only
                        the implemented ones)
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen
                         together with the prefix parameters.
        common_structure (:obj:`bool`): whether using name-based addressing with a common structure mapping.

    """
    config_class = PrefixConfig
    delta_type = "prefix"
    default_modified_modules = ['attn@']
    _supported_backends = ['hf']
    _need_pseudo_data = True
    def __init__(self,
                 backbone_model: nn.Module,
                 prefix_token_num=6,
                 reparameterize=True,
                 embed_dim: Optional[int]=512,
                 mid_dim: Optional[int]=512,
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 ):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           exclude_modules=exclude_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )

    def add_all_delta_to_backbone(self,
                 module: nn.Module,
                 modified_modules: List[str],
                ) -> nn.Module:
        first_modified_module = None
        # Current, We assume the layerer are in order in named_modules.
        # Thus the first modified module is the first module that the tensor flows to.
        for key, _ in module.named_modules():
            if self.find_key(key, modified_modules):
                logger.debug("find key {}".format(key))
                if first_modified_module is None:
                    _, _, ref = self.find_module(module, key)
                    first_modified_module = ref
                self.update_module(module, key)

        self._pseudo_data_to_instantiate(module)

        if self.reparameterize:
            reparams = ReparameterizeFunction(prefix_token_num=self.prefix_token_num,
                                              embed_dim=self.embed_dim,
                                              mid_dim=self.mid_dim,
                                              module_list=self.delta_modules)
            self.delta_modules = None
            self.reparams = reparams
            self.insert_sequential_module(first_modified_module, delta_module=reparams, delta_name="reparams", strict=False)
        self.mark_as_delta()
        return module



    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)

        prefixlayer, ref = self.new_module_like(ref)
        self.insert_sequential_module(ref, delta_module=prefixlayer, delta_name="prefix")
        self.delta_modules.append(prefixlayer)

    def new_module_like(self, module):
        # TODO: support more Attention modules

        if isinstance(module, T5Attention) or isinstance(module, T5LayerSelfAttention):
            if isinstance(module, T5LayerSelfAttention):
                module = module.SelfAttention # innermodule
            module_device = get_device(module)
            prefixlayer = PrefixLayerT5(prefix_token_num=self.prefix_token_num, num_heads=module.n_heads ,device=module_device)
        elif isinstance(module, MultiHeadSelfAttention):  # MultiHeadSelfAttention didn't provide past_key_value in the interface of the forward function.
            module_device = get_device(module)
            prefixlayer = PrefixLayerDistilBert(prefix_token_num=self.prefix_token_num, device=module_device)
            self.insert_sequential_module(getattr(module, "k_lin"), pre_caller=prefixlayer.key_pre_forward, post_caller=prefixlayer.key_forward)
            self.insert_sequential_module(getattr(module, "v_lin"), pre_caller=prefixlayer.value_pre_forward, post_caller=prefixlayer.value_forward)
        elif isinstance(module, BertAttention):
            module_device = get_device(module)
            prefixlayer = PrefixLayerBert(prefix_token_num=self.prefix_token_num, num_heads=module.self.num_attention_heads ,device=module_device)
        elif isinstance(module, RobertaAttention):
            module_device = get_device(module)
            prefixlayer = PrefixLayerRoberta(prefix_token_num=self.prefix_token_num, num_heads=module.self.num_attention_heads,device=module_device)
        elif isinstance(module, GPT2Attention):
            module_device = get_device(module)
            prefixlayer = PrefixLayerGPT2(prefix_token_num=self.prefix_token_num, num_heads=module.num_heads ,device=module_device)
        elif isinstance(module, BartAttention):
            module_device = get_device(module)
            prefixlayer = PrefixLayerBart(prefix_token_num=self.prefix_token_num, num_heads=module.num_heads ,device=module_device)
        else:
            raise NotImplementedError(f"We haven't implement Prefix Tuning Layer for {module.__class__.__name__}. Please refer to https://opendelta.readthedocs.io/en/latest/notes/faq.html for detail.")
        return prefixlayer, module












