from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
from typing import *
import torch
import torch.nn as nn
from opendelta import BaseDeltaConfig
from opendelta import logging
logger = logging.get_logger(__name__)

class SoftPromptConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`SoftPromptModel`

    """
    def __init__(
        self,
        soft_token_num=100,
        init_range = 0.5,
        token_init = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class SoftPromptLayer(nn.Module):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens.
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take
    the first n_tokens similar to their implementation).

    Note that this template can be simply achieved by :obj:`SoftManualTemplate`, in which
    you set ``n_token`` <soft> tokens template before the <text_a> will give the same result.
    """

    def __init__(self,
                 soft_token_num: int = 100,
                 raw_embedding: Optional[torch.Tensor] = None,
                 init_range: Optional[float] = 0.5,
                 other_expand_ids: Optional[Dict] = {"attention_mask":1, "token_type_ids":0},
                 token_init = False,
                 pad_id = 0,
                 device: Optional[str]=None,
                ):
        super().__init__()
        self.__dict__['raw_embedding'] = raw_embedding

        self.init_range = init_range
        self.num_tokens = soft_token_num
        self.pad_id = pad_id
        self.token_init = token_init
        self.device = device
        self.other_expand_ids = other_expand_ids

        assert self.num_tokens>0
        self.instantiate(raw_embedding(torch.tensor([0])).shape[-1])

        # self.all_pseudo_tokens = {}

    def pre_forward(self, *args, **kwargs):
        # if attention_mask is passed as PLM's input, modify it here
        if 'encoder_outputs' in kwargs and kwargs['encoder_outputs'] is not None:
            # In generation, the input is forward through the model again.
            return args, kwargs

        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
            kwargs['input_ids'] = None
        elif len(args) > 0:
            input_ids = args[0]
            args = args[1:]
        else:
            input_ids = None


        if 'attention_mask' not in kwargs or kwargs['attention_mask'] is None:
            # infer attention mask
            if input_ids is None:
                raise RuntimeError("no input ids found")
            kwargs['attention_mask'] = (input_ids != self.pad_id).to(torch.int64)

        if 'inputs_embeds' not in kwargs or kwargs['inputs_embeds'] is None:
            try:
                inputs_embeds = self.raw_embedding(input_ids)
            except:
                raise RuntimeError("neither inputs_embeds nor input_ids is specified.")
        else:
            inputs_embeds = kwargs['inputs_embeds']



        batch_size = inputs_embeds.size(0)
        soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
        inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)
        kwargs['inputs_embeds'] = inputs_embeds

        for expand_key in self.other_expand_ids:
            if expand_key in kwargs:
                real_tokens = kwargs[expand_key]
                # if expand_key in self.all_pseudo_tokens:
                #     pseudo_tokens = self.all_pseudo_tokens[expand_key].to(real_tokens.device)
                # else:
                pseudo_tokens_value = self.other_expand_ids[expand_key]
                pseudo_tokens = torch.ones(
                    (*real_tokens.shape[:-1], inputs_embeds.shape[-2]-real_tokens.shape[-1]),
                    dtype = real_tokens.dtype,
                    device=real_tokens.device) * pseudo_tokens_value
                    # self.all_pseudo_tokens[expand_key] = pseudo_tokens
                real_tokens.data = torch.cat([pseudo_tokens, real_tokens], dim=-1)

        return args, kwargs

    def instantiate(self, hidden_dim) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        soft_embeds = torch.FloatTensor(self.num_tokens, hidden_dim)
        if self.token_init:
            soft_embeds.data = torch.clone(self.raw_embedding(torch.tensor([i for i in range(self.num_tokens)])))
        else:
            soft_embeds = soft_embeds.uniform_(-self.init_range, self.init_range)

        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True).to(self.device)


class SoftPromptModel(DeltaBase):
    r"""
    This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens.
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take
    the first n_tokens similar to their implementation).

    Note that this template can be simply achieved by :obj:`SoftManualTemplate`, in which
    you set ``n_token`` <soft> tokens template before the <text_a> will give the same result.

    Args:

        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified.
        soft_token_num (:obj:`int`, *optional*): num of new tokens to add in the front of the input.
        init_range (:obj:`float`, *optional*): If initialize new tokens randomly, the random range of uniform distribution.
        token_init (:obj:`bool`, *optional*, default to :obj:`True`): Whether to initialize the new tokens with tokens of the PLM.
        other_expand_ids (:obj:`dict`, *optional*, default to ``{'attention_mask':1, 'token_type_ids':0}``): The name of other tokens and its default value that expand along with the input sequence. For example, when you prepend 100 tokens to the input_ids, the attention_mask should be extended, and the token_type_ids should be extended as well.
        modified_modules (:obj:`List[str]`): For prefix tuning, the it must refer to an attention layer (Currently, only the implemented ones).
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen together with the prefix parameters.
        common_structure (:obj:`bool`): whether using name-based addressing with a common structure mapping.


    """

    config_class = SoftPromptConfig
    delta_type = "soft_prompt"
    default_modified_modules = ["root"]  # not used
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = False
    def __init__(self,
                 backbone_model: nn.Module,
                 soft_token_num=100,
                 init_range = 0.5,
                 token_init=True,
                 other_expand_ids={"attention_mask":1, "token_type_ids":0},
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                ):
        DeltaBase.__init__(self,
                           backbone_model = backbone_model,
                           modified_modules = ["root"],
                           exclude_modules = exclude_modules,
                           unfrozen_modules = unfrozen_modules,
                           common_structure = False,
                           interactive_modify = interactive_modify,
                          )

        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])


        try:
            self.__dict__['raw_embedding'] = self.backbone_model.get_input_embeddings()
        except AttributeError:
            raise AttributeError(f"'{type(self.backbone_model)}' object has no attribute 'get_input_embeddings', please pass "+
            "input embeddings into 'self.raw_embedding' in user-specific ways.")

        self.delta_modules = nn.ModuleList()
        self.add_all_delta_to_backbone(self.backbone_model,
                                       self.modified_modules,
                                      )

    def add_all_delta_to_backbone(self,
                 module: nn.Module,
                 modified_modules: List[str],
                ) -> nn.Module:
        self.update_module()
        self.mark_as_delta()
        return module

    def update_module(self):
        soft_prompt_layer = self.new_module_like(self.raw_embedding)
        self.insert_sequential_module(self.backbone_model.get_encoder() if self.backbone_model.config.is_encoder_decoder else self.backbone_model,delta_module=soft_prompt_layer,delta_name="soft_prompt_layer"  )

    def new_module_like(self, module):
        module_device = get_device(module)
        soft_prompt_layer = SoftPromptLayer(
            soft_token_num = self.soft_token_num,
            raw_embedding = self.raw_embedding,
            other_expand_ids = self.other_expand_ids,
            token_init = self.token_init,
            init_range = self.init_range,
            device = module_device,
        )
        if self.backend == 'bmt':
            import bmtrain as bmt
            soft_prompt_layer = bmt.BMTrainModelWrapper(soft_prompt_layer)
        self.delta_modules.append(soft_prompt_layer)
        return soft_prompt_layer
