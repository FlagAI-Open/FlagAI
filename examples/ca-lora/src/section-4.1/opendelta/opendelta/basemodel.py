

from collections import OrderedDict
from multiprocessing.sharedctypes import Value
import os
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.inspect import inspect_module_statistics
from opendelta.utils.model_md5 import gen_model_hash
from opendelta.utils.signature import get_arg_names, signature
from typing import Optional, Union
from opendelta.utils.cuda import get_device
from opendelta.utils.name_based_addressing import *
import torch.nn as nn
import torch
from functools import wraps
# from decorator import decorate
from opendelta.utils.decorate import decorate
from opendelta.utils.structure_mapping import transform
from transformers.file_utils import PushToHubMixin
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from opendelta import SaveLoadMixin
from opendelta import logging
from opendelta.utils.structure_mapping import CommonStructureMap
from opendelta.utils.interactive.web import interactive
from opendelta.utils.data_parallel import new_replicate_for_data_parallel
from opendelta.utils.cuda import move_dict_to_cuda
import sys

from opendelta.utils.data_parallel import caller_map
from opendelta.utils.backend import BackendMapping
logger = logging.get_logger(__name__)

def is_leaf_module(module):
    r"""Whether the module is a leaf module
    """
    return len([n for n,_ in module.named_children()]) == 0

        

def non_module_param(module: nn.Module):
    module_names = [n for n, _ in module.named_modules()]
    ret = []
    for n, p in module.named_parameters():
        if not is_child_key(n, module_names):
            ret.append((n,p))
    return ret





class DeltaBase(nn.Module, SaveLoadMixin):
    r"""This is the base class for all delta models. It provides four simple but effective functionalities
    for building the delta model:

        #. addressing a module inside the backbone model using a minimal description key.
        #. provide the interface for modifying and inserting model which keeps the docs/IO the same as the module
           before modification.
        #. pass a pseudo input to determine the inter dimension of the delta models.
        #. freeze a part of model parameters according to key.

        It also provides unified interface for model loading and saving.

    Class attributes (overridden by derived classes):

        - delta_type (:obj:`str`): the name of the delta modules, used to create the correct :class:`opendelta.AutoDeltaModel`.
        - config_class (:class:`BaseDeltaConfig`): The corresponding config model


    Args:
        backbone_model (:obj:`nn.Module`, *required*):  backbone model that the delta models are build opon. The modification to the
            backbone model are in place.
        modified_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules are subjected to update.

            .. note::
                leave this argument :obj:`None` will make the delta model return to the default setting, which add the delta
                models to the position experimented the paper. In this setting, the common structure mapping is loaded to
                addressing the corresponding modules.
        exclude_modules (:obj:`str`, *optional*, default to :obj:`None`): The modules starts with these strings will be excluded in modification.
                Note that currently only plain text (no regular expression) is supported.
        unfrozen_modules (:obj:`str`, *optional*, default to :obj:`None`): The modules that are **not** frozen when freezing the main part of the model.
        registraction_name (:obj:`str`, *optional*, default to ``"deltas"``): The root name of the delta models when
            attached to the backbone model.
        common_structure (:obj:`bool`, *optional*, default to :obj:`None`): Whether use the common structure mapping to specify the
                modified_modules. i.e., if common_structure=True, then we use a common ["attn"] for attention module in different models.
                We DO NOT recommend manually set ``common_structure`` to ``true`` by yourself unless you are using delta
                among multiple backbones and don't want to modify the code.

        interactive_modify (:obj:`bool` or :obj:`int`, *optional*, default to :obj:`None`): Whether to use interactive modification.
            By setting to :obj:`int` can specify the port of web server.
    """
    delta_type = ""
    default_modified_modules = []
    default_exclude_modules = ["lm_head"]
    config_class = BaseDeltaConfig
    default_unfrozen_modules = ["deltas"]
    _need_pseudo_data = True
    _supported_backends = ['hf']
    def __init__(self,
                 backbone_model: nn.Module,
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 common_structure: Optional[bool] = False,
                 backend: Optional[str]= "hf", # select from ["hf", "bmt"]
                 ):
        nn.Module.__init__(self)
        # register the backbone model after init using self.__dict__ method to avoid adding backbone_model
        # to the modules of the delta model.
        self.__dict__["backbone_model"] = backbone_model
        if modified_modules is None and exclude_modules is None:
            if interactive_modify:
                if isinstance(interactive_modify, bool) and interactive_modify==True:
                    self.modified_modules = interactive(backbone_model)
                else:
                    self.modified_modules = interactive(backbone_model, port=interactive_modify)
                self.common_structure = False
                self.exclude_modules = self.default_exclude_modules
            else:
                self.modified_modules = self.default_modified_modules
                self.common_structure = True
                self.exclude_modules = self.default_exclude_modules
        else:
            if interactive_modify:
                raise ValueError("Use modified_modules(or exclude modules) and interactive_modify at the same time is not supported")
            if modified_modules is not None:
                self.modified_modules = modified_modules
            else:
                self.modified_modules = self.default_modified_modules
            if exclude_modules is not None:
                self.exclude_modules = exclude_modules
            else:
                self.exclude_modules = self.default_exclude_modules
            self.common_structure = common_structure
        if self.common_structure:
            self.structure_mapping = CommonStructureMap(self.backbone_model)
        else:
            self.structure_mapping = None
        if unfrozen_modules is None:
            self.unfrozen_modules = self.default_unfrozen_modules
        else:
            self.unfrozen_modules = unfrozen_modules
        if self.common_structure and self.structure_mapping is None:
            raise RuntimeError("Using common structure but the structure mapping is None")
        if backend not in self._supported_backends:
            raise RuntimeError("Currently, backend `{}` is not supported for `{}`".format(backend, self.__class__.__name__))
        self.backend = backend
        self.backend_mapping = BackendMapping(backend)

    def forward(self, *args, **kwargs) -> RuntimeError:
        r"""
            .. warning::

                Removed method. As the model is a delta model, which should be attached to a backbone model \
                and can't forward any data by itself. Please using the backbone model's forward function \
                after attach the delta model to the backbone.
        """
        raise RuntimeError("This is a delta model, which should be attached to a backbone model \
            and can't forward any data by itself. Please using the backbone model's forward function \
            after attach the delta model to the backbone. ")

    @classmethod
    def from_config(cls, config: Union[BaseDeltaConfig, dict], backbone_model: nn.Module, check_hash=True, **kwargs):
        r"""Initialize a delta model from a config object or a dict containing the configs. To temperarily change
        a value in the config, pass it through kwargs. If the config has a backbone model's hash, which means it is
        a finetuned delta model's config, then we will compare the hash in the config and the newly caculated to ensure
        the finedtuned delta model is trained on the passed backbone_model. Pass ``check_hash=False`` to disable the
        checking.

        Args:
            config (:obj:`BaseDeltaConfig` or :obj:`dict`) A config object or a dict that contains the necessary value to
                            initialize the delta model.
            backbone_model (:obj:`nn.Module`) A pytorch module that will be pass into the delta model as the backbone
                    model. modifications will be made in place in the backbone model.
            check_hash (:obj:`bool`, default to ``True``) Whether to check hash of the backbone model and the config's
                            backbone hash.
            kwargs: Any configurations that are passed to update the config object. #TODO unit test needed.
        """
        supported_keys = get_arg_names(cls.__init__) + get_arg_names(DeltaBase.__init__)
        config_dict = config.to_dict()
        for key in list(config_dict.keys()):
            if key not in supported_keys:
                config_dict.pop(key)
        return cls(backbone_model, **config_dict)


    def add_all_delta_to_backbone(self,
                 backbone: nn.Module,
                 modified_modules: List[str],
                ) -> nn.Module:
        r"""The main function to add delta models to the backbone model based on the :obj:`modified_modules`.


        Args:
            backbone_model (:obj:`nn.Module`, *required*)  backbone model that the delta models are build opon. The
                modification to the backbone model are in place.
            modified_modules (:obj:`List[str]`, *optional*, default to :obj:`None`) The modules are subjected to update.
                leave this argument :obj:`None` will make the delta model return to the default setting, which add the delta
                models to the position experimented the paper. In this setting, the common structure mapping is loaded to
                addressing the corresponding modules.

        Returns:
            :obj:`nn.Module` The modified backbone model.

        """
        self.plm_total_params = sum(p.numel() for p in backbone.parameters())
        # create a new key list to avoid recursion.
        backbone_key_list = [key for key, _ in backbone.named_modules()]
        for key in backbone_key_list:
            if self.find_key(key, modified_modules):
                self.update_module(backbone, key)
        if self._need_pseudo_data:
            self._pseudo_data_to_instantiate(backbone)
                    
        # mark the paratmers that are the delta parameters for easily displaying the delta_paramters.
        self.mark_as_delta()
        return backbone

    def _pseudo_data_to_instantiate(self, backbone: Optional[nn.Module]=None):
        if self.structure_mapping is None:
            self._pseudo_data_to_instantiate_module(backbone)
        else:
            for key in self.structure_mapping.matched_pairs:
                if key == "":
                    submodule = backbone
                else:
                    _, _, submodule = self.find_module(backbone, key)
                self._pseudo_data_to_instantiate_module(submodule)

    def mark_as_delta(self, module: nn.Module=None,):
        r"""[NODOC] Mark :obj:`module`'s all parameters as delta parameters by setting a ``_is_delta`` attribute to each of them.
        Generally, it is used after creating the delta modules. By leaving module to :obj:`None`, it will mark all the parameters in the
        delta model as ``_is_delta``.

        Args:
            module (:obj:`nn.Module`): The module to mark as delta.
        """
        if module is None:
            module=self # all the parameters in the delta model.
        for p in module.parameters():
            setattr(p, "_is_delta", True)

    def update_module(self, module: nn.Module, key: str):
        r"""Update a module specified by :obj:`key`. The method is reimplemented in each specific delta model.
        """
        raise NotImplementedError


    def freeze_module(self,
                      module: Optional[nn.Module] = None,
                      exclude: Optional[List[str]] = None,
                      set_state_dict: Optional[bool]=True,
                      ):
        r"""Freeze the parameters of plm. Leave the parameters in exclude untouched.
        deltas module is filtered with ``_is_delta`` attributes because it may have parameter sharing to the main
        model, (e.g., bias term)

        Args:
            module (:obj:`nn.Module`, *optional*, default to :obj:`None`): The module of which some parts are frozen.
                If left with :obj:`None`, the function will the self.backbone_model as the module to be frozen.
            exclude (:obj:`List[str]`, *optional*, default to ``["deltas"]``): The parameters that don't need to
                be freezed. Default to all the delta parameters.
            set_state_dict (:obj:`bool`, *optional*, default to :obj:`True`): Whether setting the backbone model's state
                dict to all the parameters that still need grad.
            prefix (:obj:`str`, *optional*, default to ``""``): A parameters that are used for recursive frozen.
                Should not be changed by passing argument other than ``""``.

        """
        if exclude is None:
            exclude = self.unfrozen_modules

        if module is None:
            module = self.backbone_model
        self._freeze_module_recursive(module, exclude, "")    # modify the active state dict that still need grad
        if set_state_dict:
            self.set_active_state_dict(module)

    def _freeze_module_recursive(self,
                      module: Optional[nn.Module] = None,
                      exclude: Optional[List[str]] = None,
                      prefix=""):
        r"""[NODOC] Freeze the parameters of plm. Leave the parameters in exclude untouched.
        deltas module is filtered with ``_is_delta`` attributes because it may have parameter sharing to the main
        model, (e.g., bias term)

        Args:
            module (:obj:`nn.Module`, *optional*, default to :obj:`None`): The module of which some parts are frozen.
                If left with :obj:`None`, the function will the self.backbone_model as the module to be frozen.
            exclude (:obj:`List[str]`, *optional*, default to ``["deltas"]``): The parameters that don't need to
                be freezed. Default to all the delta parameters.
            set_state_dict (:obj:`bool`, *optional*, default to :obj:`True`): Whether setting the backbone model's state
                dict to all the parameters that still need grad.
            prefix (:obj:`str`, *optional*, default to ``""``): A parameters that are used for recursive frozen.
                Should not be changed by passing argument other than ``""``.

        """

        if is_leaf_module(module):
            for n, p in module.named_parameters():
                next_prefix = n if prefix == "" else ".".join([prefix,n])
                if self.find_key(next_prefix, exclude):
                    continue
                if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                    p.requires_grad = False
            return
        else:
            # firstly freeze the non module params, then go deeper.
            params = non_module_param(module)
            for n, p in params:
                if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                    p.requires_grad = False
            for n, c in module.named_children():
                next_prefix = n if prefix == "" else ".".join([prefix,n])
                if self.find_key(next_prefix, exclude): # if found, untouch the parameters
                    continue
                else:
                    self._freeze_module_recursive(c, exclude=exclude, prefix=next_prefix)





    def find_key(self, key: str, target_list: List[str]):
        r"""Check whether any target string is in the key or in the tail of the key, i.e.,

        Args:
            key (:obj:`str`): The key (name) of a submodule in a ancestor module.
                                 E.g., model.encoder.layer.0.attention
            target_list (List[Union[:obj:`str`, :obj:`re.Pattern`]]): The target list that we try to match ``key`` with. E.g., ["attention"]

        Returns:
            :obj:`bool` True if the key matchs the target list.
        """
        for x in self.exclude_modules:
            if key.startswith(x): # start with the excluded key
                return False
        virtual_key, in_virtual_order = None, None
        if self.structure_mapping is not None:
            key, virtual_key, in_virtual_order = self.structure_mapping.transform(key, strict=False)
            # currently in_virtual_order not in use, it means that if the common structure designate adding adapter to FFN, it will be add to all submodule of FFN. 
        if not key:
            return False
        if virtual_key is None:
            return endswith_in(key, target_list)
        else:
            return endswith_in(key, target_list) or endswith_in(virtual_key, target_list)


    def _pseudo_data_to_instantiate_module(self, module: Optional[nn.Module]=None):
        r"""Some delta model requires a pseudo-data be passed through the model to understand the dimensionality of each tensor in the computation graph.

        (1) The model in the Huggingface Transformers library usually has the so-called `dummy_inputs`. We will make use of it.
        (2) If the model does not have `dummy_inputs`, we will try to create it and throw a warning.
        (3) If we encounter an error in (2), we will suggest you to create it by passing the dummy_inputs variable.

        Args:
            module (:obj:`nn.Module`, *optional*, default to :obj:`None`): The backbone model.

        """
        if module is None:
            module = self.backbone_model
        device = get_device(module)
        _auto_dummy = False
        try:
            dummy_inputs = module.dummy_inputs
            dummy_inputs = move_dict_to_cuda(dummy_inputs, device)
        except AttributeError:
            logger.warning(f"No `dummy_inputs` attribute in {module.__class__.__name__} , automatically create `dummy_inputs`. Very likely to encounter error. To set dummy_inputs for your model, please use: `setattr(backbone_model, 'dummy_inputs', some_dummy_inputs)` before initializing `{self.__class__.__name__}`")
            _auto_dummy = True
            pass
        if _auto_dummy:
            _most_simple_input = torch.tensor([[0,0]]).to(device)
            if "decoder_input_ids" in  signature(module.forward).args:
                dummy_inputs = {"input_ids": _most_simple_input, "decoder_input_ids": _most_simple_input}
            else:
                dummy_inputs = {"input_ids": _most_simple_input}

        _auto_dummy_fail = False
        try:
            module(**dummy_inputs)
        except Exception as e:
            _auto_dummy_fail = True
        
            if _auto_dummy_fail and _auto_dummy:  
                raise AttributeError(f"str({e})\n\tThe {self.__class__.__name__} requires a dummy_inputs to be passed through the model to understand the dimensionality of each tensor in the computation graph. \n\t The {module.__class__.__name__} Class has no dummy_inputs, and automatically created dummy_inputs failed.\n\t Refer to `https://opendelta.readthedocs.io/en/latest/notes/faq.html` for detail.")
           




    def trainable_parameters_names(self, module: Optional[nn.Module]=None):
        r"""[NODOC] A small sugar function to return all the trainable parameter's name in the (by default, backbone) model.

        Args:
            module (:obj:`nn.Module`): of which module we want to know the trainable paramemters' name.

        Returns:
            :obj:`List[str]`
        """
        if module is None:
            module = self.backbone_model
        return [n for n,p in module.named_parameters() if p.requires_grad]

    def frozen_parameters_names(self, module: Optional[nn.Module]=None):
        r"""[NODOC] A small sugar function to return all the frozen parameters' name in the (by default, backbone) model.

        Args:
            module (:obj:`nn.Module`): of which module we want to know the frozen paramemters' name.

        Returns:
            :obj:`List[str]`
        """
        if module is None:
            module = self.backbone_model
        return [n for n,p in module.named_parameters() if not p.requires_grad]

    def trainable_parameters(self,module: Optional[nn.Module]=None):
        r"""[NODOC] A small sugar function to return all the frozen parameters in the (by default, backbone) model.

        Args:
            module (:obj:`nn.Module`): of which module we want to know the frozen paramemters.

        Returns:
            :obj:`List[nn.Parameter]`
        """
        if module is None:
            module = self
        return [p for n,p in module.named_parameters() if p.requires_grad]


    def num_trainable_parameters(self, module: Optional[nn.Module]=None):
        r"""[NODOC] A small sugar function to get the number of trainable parameter in the backbone model. Often used to
        compute the trainable rate.

        Args:
            module (:obj:`nn.Module`): of which module we want to know the number of trainable paramemters.

        Returns:
            :obj:`List[nn.Parameter]`
        """
        if module is None:
            module = self
        pnum_tot = 0
        for param in module.parameters():
            if param.requires_grad:
                pnum_tot += param.numel()
        return pnum_tot

    def num_total_parameters(self, module: Optional[nn.Module]=None):
        r"""[NODOC] A small sugar function to get the number of trainable parameter in the backbone model. Often used to
        compute the trainable rate.

        Args:
            module (:obj:`nn.Module`): of which module we want to know the number of trainable paramemters.

        Returns:
            :obj:`List[nn.Parameter]`
        """
        if module is None:
            module = self
        pnum_tot = 0
        for param in module.parameters():
            pnum_tot += param.numel()
        return pnum_tot



    def find_module(self, root_module: nn.Module, key:str):
        r"""Find the module using a key and the root module. Return both the parent reference, the child name and reference.

        Args:
            root_module (:obj:`root_module`): The root_module to find the sub module in
            key (:obj:`str`): The relative key to the root module.

        Returns:
            (:obj:`nn.Module`, :obj:`str`, :obj:`nn.Module`):
            * A reference to the parent module of the target module, mainly for substuting the target module.
            * The key of the target module relevant to its parent module
            * Target module.
        """
        sub_keys = key.split(".")
        parent_module = root_module
        for sub_key in sub_keys[:-1]:
            parent_module = getattr(parent_module, sub_key)
        module = getattr(parent_module, sub_keys[-1])
        return parent_module, sub_keys[-1], module

    def _register_delta_infos(self, parent_module, _delta_info):
        r"""Register the delta infomation.
        Automatically incrementing the suffix for repeated delta_names
        """
        _delta_infos = getattr(parent_module, "_delta_infos", [])
        if len(_delta_infos) > 0: # check if duplicated name
            list_of_deltas = [d['delta_name'] for d in _delta_infos]
            cur_name = _delta_info['delta_name']
            if cur_name in list_of_deltas:
                cur_name = cur_name + "_1"
            counter = 1
            while cur_name in list_of_deltas:
                counter += 1
                cur_name = cur_name.split("_")[0] + "_"+str(counter)
            _delta_info["delta_name"] = cur_name
        _delta_infos.append(_delta_info)
        setattr(parent_module, "_delta_infos", _delta_infos)

    def replace_module(self,
                      parent_module: nn.Module,
                      child_name: str,
                      child_module: nn.Module,
                      new_module: nn.Module,
                      delta_name: Optional[str] = "delta",
                      ):
        r"""Replace a module's child module with the new_module(a delta module). Used by delta method based on direct
        replacement, such as :class:`opendelta.delta_modules.lora.LoraModel`.

        Args:
            parent_module (:obj:`nn.Module`): The parent module of the replacement.
            child_name (:obj:`str`): The chird module's name, i.e., parent_module.child_name give us child_module
            child_module (:obj:`nn.Module`): The original child module.
            new_module (:obj:`nn.Module`): The delta module.
            delta_name (:obj:`str`, *optional*, default ot ``delta``): The name of the delta module, used for recording.
                            parent_module.delta_name WILL NOT give you the delta module.
        """
        self.delta_modules.append(new_module)
        setattr(parent_module, child_name, new_module)
        # register delta info
        _delta_info = {"method": "replace",
                      "delta_module": new_module,
                      "child_name": child_name,
                      "org_module": child_module,
                      "delta_name": delta_name,
                      "delta_belong": self,
                      "state": "on"}
        self._register_delta_infos(parent_module=parent_module,
                                   _delta_info = _delta_info,
                                  )


    def modify_module(self, module: nn.Module):
        r"""Modify the inside parameteres of a module. This method will be reimplemented in different
        derived class if needed.
        """
        raise NotImplementedError

    def insert_module(self, module, method='sequential', delta_module=None, delta_name='delta', strict=False, _delta_info=None):
        r"""insert a module (previous not exists in the code base) before/after a module. Specifically, it modifies the forward
        function of the original module to  firstly pass the arguments into the new module's forward function and then pass
        it into the original ones. The new module can also be inserted after the original module with similar mechanism.

        When implementing the new module , researchers should be aware of the components of arguments of the original module's forward function.

        Args:
            module: (:obj:`nn.Module`): The (sub)module to inserted a delta module.
            delta_module: (:obj:`DeltaBase`): The delta module to be inserted.
            name: (:obj:`str`, *optional*): The name of the delta in the backbone module.
            strict: (:obj:`bool`, *optional*): Whether to prohibit modify a modified module.
            _delta_info (:obj:`Dict`, *optional*): Used in attach(), reattach a delta module to backbone. The info of
                                    original delta is passed through ``_delta_info``.

        """


        if strict:
            if hasattr(module.forward, "__wrapped__"):
                raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended?")

        # record info for plug and unplug and nested wrap
        if _delta_info is None:
            if delta_module is None:
                raise RuntimeError("delta module can't be none to ensure successful replicate of the parent module.")
        
            _delta_info = {"method": method,
                        "delta_module": delta_module, 
                        "delta_name": delta_name,
                        "delta_belong": self,
                        "state": "on"}
            self._register_delta_infos(parent_module=module,
                                    _delta_info = _delta_info)
        else:
            delta_module = _delta_info["delta_module"]
            delta_name = _delta_info["delta_name"]

        setattr(module, _delta_info['delta_name'], _delta_info["delta_module"])


        if _delta_info["method"] in caller_map.keys():
            caller = caller_map[_delta_info["method"]]
            new_forward = decorate(module.forward, caller, extras=(module, _delta_info['delta_name']), kwsyntax=True) # decorator.decorate helps preserving the functions metadata (signature, etc.).
            module.forward = new_forward.__get__(module, type(module))  # func.__get__(object, type(object)) register a function as an object's method
            # for DataParallel's copy behavior. Experimental:
            # may have bugs when module.forward is nestedly wrapped.
            module._replicate_for_data_parallel = new_replicate_for_data_parallel.__get__(module, type(module)) 
        else:
            raise NotImplementedError(f"_delta_info['method']=='{_delta_info['method']}' is not supported")


    def insert_sequential_module(self, module, delta_module=None, delta_name='delta', strict=False, _delta_info=None):
        r"""insert a module (previous not exists in the code base) before/after a module. Specifically, it modifies the forward 
        function of the original module to  firstly pass the arguments into the new module's forward function and then pass
        it into the original ones. The new module can also be inserted after the original module with similar mechanism. 

        When implementing the new module , researchers should be aware of the components of arguments of the original module's forward function.
        
        Args:
            module: (:obj:`nn.Module`): The (sub)module to inserted a delta module.
            delta_module: (:obj:`DeltaBase`): The delta module to be inserted.
            name: (:obj:`str`, *optional*): The name of the delta in the backbone module.
            strict: (:obj:`bool`, *optional*): Whether to prohibit modify a modified module.
            _delta_info (:obj:`Dict`, *optional*): Used in attach(), reattach a delta module to backbone. The info of 
                                    original delta is passed through ``_delta_info``.
        
        """
        self.insert_module(module, "sequential", delta_module, delta_name, strict, _delta_info)
                                             

    def insert_parallel_module(self, module, delta_module=None, delta_name='delta', strict=False, _delta_info=None):
        """insert a module (previous not exists in the code base) across a module. Specifically, it modifies the forward
        function of the original module to  firstly pass the arguments into the delta model's forward function and set
        aside the calculation result. Then combine it with the calculation result output from the backbone module.

        When implementing the new module , researchers should be aware of the arguments and keywards of the original module's forward function.

        Args:
            module: (:obj:`nn.Module`): The (sub)module to inserted a delta module.
            delta_module: (:obj:`DeltaBase`): The delta module to be inserted.
            name: (:obj:`str`, *optional*): The name of the delta in the backbone module.
            strict: (:obj:`bool`, *optional*): Whether to prohibit modify a modified module.
            _delta_info (:obj:`Dict`, *optional*): Used in attach(), reattach a delta module to backbone. The info of
                                    original delta is passed through ``_delta_info``.

        """

        self.insert_module(module, "parallel", delta_module, delta_name, strict, _delta_info)
        

    def set_active_state_dict(self, module: nn.Module):
        r"""modify the state_dict function of the model (by default, the backbone model) to return only the tunable part.

        Args:
            module (:obj:`nn.Module`): The module modified. The modification is in-place.
        """
        def _caller(_org_func, includes,  *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n  in keys:
                if n not in includes:
                    state_dict.pop(n)
            return state_dict
        includes = self.trainable_parameters_names(module) # use excludes will have trouble when the model have shared weights
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended? Do you freeze the parameters twice?")
        module.state_dict = decorate(module.state_dict, _caller, extras=(includes,), kwsyntax=True) # decorator.decorate helps preserving the functions metadata (signature, etc.).

    def _load_state_dict_into_backbone(self, backbone_model: nn.Module = None, state_dict: dict = {}):
        r"""[NODOC]
        """
        if backbone_model is None:
            backbone_model = self.backbone_model
        self.backbone_model.load_state_dict(state_dict, strict=False)

    def create_config_from_model(self, ):
        r"""[NODOC] If the delta model was built by directly passing arguments, instead of passing a config object.
        create the config of the delta model for saving the delta model.
        """
        # common_attributes
        config = self.config_class()
        config_keys = signature(config.__init__)[0] + signature(super(self.config_class, config).__init__)[0]

        for key in config_keys:
            val = getattr(self, key) if hasattr(self, key) else None
            setattr(config, key, val)
        config.delta_type = self.delta_type
        self.config = config


    def log(self, module=None, delta_ratio=True, trainable_ratio=True, visualization=True, cuda_memory=True):
        r"""Log and visualize the result of applying delta.
        Possible Options are ``trainable_ratio``,
        ``visualization``, ``delta_ratio``.

        Args:
            delta_ratio (:obj:`bool`, *optional*): Whether computing the ratio of parameters in the delta modules.
            trainable_ratio (:obj:`bool`, *optional*): Whether computing the ratio of trainable parameters.
            visualization (:obj:`bool`, *optional*): Whether visualize the parameter information of the modified backbone.

        """
        if module is None:
            module = self.backbone_model


        if visualization:
            from bigmodelvis import Visualization
            Visualization(module).structure_graph()

        self.stat = inspect_module_statistics(module, verbose=False)
        if trainable_ratio:
            logger.info("Trainable Ratio: {}/{}={:.6f}%".format(self.stat['trainable_parameters'], self.stat['total_parameters'], self.stat['trainable_ratio']*100))
        if delta_ratio:
            logger.info("Delta Parameter Ratio: {}/{}={:.6f}%".format(self.stat['delta_parameters'], self.stat['total_parameters'],self.stat['delta_ratio']*100))
        if cuda_memory:
            logger.info("Static Memory {:.2f} GB, Max Memory {:.2f} GB".format(self.stat['cudamem'], self.stat['maxcudamem']))




    # Two functions for plug and remove the delta model.
    def attach(self, module: Optional[nn.Module]=None, reset_state_dict=True):
        r"""Reattach the delta modules to the backbone. Note that this method can not be used to create new delta modules.
        Instead, a :meth:`DeltaBase.detach` should precede this method.

        Args:
            module (:obj:`object`, *optional*, default to :obj:`None`): The backbone module that we
                                                    reattach the deltas to.
        """

        if module is None:
            module = self.backbone_model

        for name, submodule in module.named_modules():
            if hasattr(submodule, "_delta_infos"):
                _delta_infos = getattr(submodule, "_delta_infos")
                for _delta_info in _delta_infos:
                    if _delta_info['delta_belong'] is not self:
                        continue
                    if _delta_info["state"] == "on":
                        continue

                    if _delta_info['method'] == "replace":
                        setattr(submodule, _delta_info["child_name"], _delta_info['delta_module'])
                    elif _delta_info['method'] == "insert_sequential":
                        self.insert_sequential_module(module=submodule,
                                    _delta_info=_delta_info)
                    elif _delta_info['method'] == "insert_parallel":
                        self.insert_parallel_module(module=submodule,
                                    _delta_info=_delta_info)
                    else:
                        raise NotImplementedError

                    _delta_info['state'] = "on"
        if reset_state_dict:
            self.set_active_state_dict(module)



    def detach(self, module: Optional[nn.Module]=None, reset_state_dict=True):
        r"""Detach the delta module from the backbone. The delta module is not deleted, but temporarily turned off.
        Use :meth:`DeltaBase.attach` to reattach the delta model to the backbone.

        Args:
            module (:obj:`object`, *optional*, default to :obj:`None`): The backbone module that we
                                                    detached the deltas from.
        """

        if module is None:
            module = self.backbone_model

        for name, submodule in module.named_modules():
            if hasattr(submodule, "_delta_infos"):
                _delta_infos = getattr(submodule, "_delta_infos")
                for _delta_info in _delta_infos:
                    if _delta_info['delta_belong'] is not self:
                        continue
                    if _delta_info["state"] == "off":
                        continue

                    if _delta_info['method'] == "replace":
                        setattr(submodule, _delta_info["child_name"], _delta_info['org_module'])
                    elif _delta_info['method'] in ["sequential", "before", "after", "parallel"]:
                        if hasattr(submodule.forward, "__wrapped__"):
                            submodule.forward = submodule.forward.__wrapped__
                            delattr(submodule, _delta_info["delta_name"])
                        else:
                            raise AttributeError("submodule {}'s forward has no attribute __wrapped__. It's not a wrapped function.".format(name))
                    else:
                        raise NotImplementedError

                    _delta_info['state'] = "off"
        if reset_state_dict:
            try:
                module.state_dict = module.state_dict.__wrapped__
            except AttributeError:
                pass

