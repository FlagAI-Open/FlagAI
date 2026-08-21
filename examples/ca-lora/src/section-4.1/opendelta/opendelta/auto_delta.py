from copy import deepcopy
from typing import Any, Dict, OrderedDict
from bigmodelvis import Visualization
import torch.nn as nn
from opendelta.utils.logging import get_logger
import importlib
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.basemodel import DeltaBase
logger = get_logger(__name__)


DELTA_CONFIG_MAPPING = {
    "lora": "LoraConfig",
    "low_rank_adapter": "LowRankAdapterConfig",
    "bitfit": "BitFitConfig",
    "adapter":"AdapterConfig",
    "compacter":"CompacterConfig",
    "prefix": "PrefixConfig",
    "soft_prompt": "SoftPromptConfig",
    "parallel_adapter": "ParallelAdapterConfig",
}

DELTA_MODEL_MAPPING = {
    "lora": "LoraModel",
    "low_rank_adapter": "LowRankAdapterModel",
    "bitfit": "BitFitModel",
    "adapter":"AdapterModel",
    "compacter": "CompacterModel",
    "prefix": "PrefixModel",
    "soft_prompt": "SoftPromptModel",
    "parallel_adapter": "ParallelAdapterModel",
}

class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = key #model_type_to_module_name(key)
        # if module_name not in self._modules:
        self._modules[module_name] = importlib.import_module(f".{module_name}", "opendelta.delta_models")
        return getattr(self._modules[module_name], value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys():
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


LAZY_CONFIG_MAPPING = _LazyConfigMapping(DELTA_CONFIG_MAPPING)



class AutoDeltaConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the :meth:`~AutoDeltaConfig.from_finetuned` or :meth:`~AutoDeltaConfig.from_dict` class method. 
    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self, *args, **kwargs):
        raise AttributeError(
            f"{self.__class__.__name__} is designed to be instantiated using\n\t(1) `{self.__class__.__name__}.from_finetuned(finetuned_model_name_or_path)`\nor\t(2) `{self.__class__.__name__}.from_dict(config_dict, **kwargs)` "
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        r""" Instantiate a DeltaConfig according to the dict. Automatically load the config specified by
        :obj:`delta_type`.

        Args:
            config_dict (:obj:`dict`): The dict of configs of delta model.
            kwargs: Other keyword argument pass to initialize the config.

        Examples:
        
        .. code-block:: python

            config = AutoDeltaConfig.from_dict({"delta_type":"lora"}) # This will load the dault lora config.
            config = AutoDeltaConfig.from_dict({"delta_type":"lora", "lora_r":5}) # Will load the default lora config, with lora_r = 5

        """
        config_dict = deepcopy(config_dict)
        delta_type = config_dict.pop("delta_type", None)
        if delta_type is None:
            raise RuntimeError("Do not specify a delta type, cannot load the default config")
        config_class = LAZY_CONFIG_MAPPING[delta_type]
        return config_class.from_dict(config_dict, **kwargs)


    @classmethod
    def from_finetuned(cls, finetuned_delta_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a finetuned delta model configuration.
        The configuration class to instantiate is selected based on the ``delta_type`` property of the config object that
        is loaded.

        Parameters:

            finetuned_delta_path (:obj:`str` or :obj:`os.PathLike`, *optional*): Can be either:

                - A string, the model id of a finetuned delta model configuration hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like ``Davin/lora``, or namespaced under a user or organization name, like ``DeltaHub/lora_t5-base_mrpc``.
                - A path to a *directory* containing a configuration file saved using the :py:meth:`~opendelta.basemodel.DeltaBase.save_finetuned` method, e.g., ``./my_model_directory/``.
                - A path or url to a saved configuration JSON *file*, e.g.,``./my_model_directory/configuration.json``.

            cache_dir (:obj:`str` or :obj:`os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            
        Examples:

        .. code-block:: python

            from transformers import AutoConfig
            delta_config = AutoDeltaConfig.from_finetuned("thunlp/FactQA_T5-large_Adapter")

        """


        config_dict, kwargs = BaseDeltaConfig.get_config_dict(finetuned_delta_path, **kwargs)
        if "delta_type" in config_dict:
            config_class = LAZY_CONFIG_MAPPING[config_dict["delta_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in LAZY_CONFIG_MAPPING.items():
                if pattern in str(finetuned_delta_path):
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            f"Unrecognized model in {finetuned_delta_path}. "
            f"Should have a `delta_type` key in the loaded config, or contain one of the following strings "
            f"in its name: {', '.join(LAZY_CONFIG_MAPPING.keys())}"
        )

### AutoModels below

class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:

        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type not in self._model_mapping:
            raise KeyError(key)
        model_name = self._model_mapping[model_type]
        return self._load_attr_from_module(model_type, model_name)

    def _load_attr_from_module(self, model_type, attr):
        if model_type not in self._modules:
            self._modules[model_type] = importlib.import_module(f".{model_type}", "opendelta.delta_models")
        return getattribute_from_module(self._modules[model_type], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys():
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        self._extra_content[key] = value



LAZY_DELTA_MAPPING = _LazyAutoMapping(DELTA_CONFIG_MAPPING, DELTA_MODEL_MAPPING)



def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("transformers")
    return getattribute_from_module(transformers_module, attr)



class AutoDeltaModel:
    r"""
    """
    _delta_model_mapping = LAZY_DELTA_MAPPING
    def __init__(self, *args, **kwargs):
        # raise EnvironmentError(
        #     f"{self.__class__.__name__} is designed to be instantiated "
        #     f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
        #     f"`{self.__class__.__name__}.from_config(config)` methods."
        # )

        raise AttributeError(
            f"{self.__class__.__name__} is designed to be instantiated using\n\t(1) `{self.__class__.__name__}.from_finetuned(finetuned_delta_path, backbone_model, *model_args, **kwargs)`\nor\t(2) `{self.__class__.__name__}.from_config(delta_config, backbone_model, **kwargs)`"
        )

    @classmethod
    def from_config(cls, config, backbone_model, **kwargs) -> DeltaBase:
        r"""Automatically instantiates a delta model based on the :obj:`config`. The delta model correspond to the delta
        :obj:`config` will be loaded and initialized using the arguments in :obj:`config`.

        .. note::
            Only using :meth:`from_config` method will not load the finetuned weight file (e.g., pytorch_model.bin).
            Please use from_finetuned directly.

        Args:
            config (:obj:`BaseDeltaConfig`):
            backbone_model (:obj:`nn.Module`):

        Examples:

        .. code-block:: python

            config = AutoDeltaConfig.from_finetuned("DeltaHub/lora_t5-base_mrpc")
            delta_model = AutoDeltaModel.from_config(config, backbone_model)

        """
        if type(config) in cls._delta_model_mapping.keys():
            model_class = cls._delta_model_mapping[type(config)]
            return model_class.from_config(config, backbone_model, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._delta_model_mapping.keys())}."
        )

    @classmethod
    def from_finetuned(cls, finetuned_delta_path, backbone_model, *model_args, **kwargs) -> DeltaBase:
        r""" Automatically instantiated a delta model and load the finetuned checkpoints based on the
        :obj:`finetuned_delta_path`, which can either be a string pointing to a local path or a url pointint to
        the delta hub. It will check the hash after loading the delta model to see whether the correct backbone and
        delta checkpoint are used.

        Args:
            finetuned_delta_path (:obj:`str` or :obj:`os.PathLike`, *optional*): Can be either: 

                - A string, the model name of a finetuned delta model configuration hosted inside a model repo on `Delta Center <https://www.openbmb.org/toolKits/deltacenter>`_, like ``thunlp/FactQA_T5-large_Adapter``.
                - A path to a directory containing a configuration file saved using the :meth:`~opendelta.utils.saving_loading_utils.SaveLoadMixin.save_finetuned` method, e.g., ``./my_model_directory/``.
                - A path or url to a saved configuration JSON *file*, e.g., ``./my_model_directory/configuration.json``.The last two option are not tested but inherited from huggingface.

            backbone_model (:obj:`nn.Module`): The backbone model to be modified.
            model_args: Other argument for initialize the model. See :`DeltaBase.from_finetuned` for details.
            kwargs: Other kwargs that will be passed into DeltaBase.from_finetuned. See `DeltaBase.from_finetuned` for details.

        Example:

        .. code-block:: python

            delta_model = AutoDeltaModel.from_finetuned("thunlp/FactQA_T5-large_Adapter", backbone_model=5)

        """
        delta_config = kwargs.pop("delta_config", None)

        if not isinstance(delta_config, BaseDeltaConfig):
            delta_config, kwargs = AutoDeltaConfig.from_finetuned(
                finetuned_delta_path, return_unused_kwargs=True, **kwargs
            )
        if type(delta_config) in cls._delta_model_mapping.keys():
            model_class = cls._delta_model_mapping[type(delta_config)]
            return model_class.from_finetuned(finetuned_delta_path, backbone_model, *model_args, delta_config=delta_config,  **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )





if __name__ == "__main__":

    config = AutoDeltaConfig.from_dict({"delta_type":"lora", "lora_r": 7})


    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("../../plm_cache/roberta-base/", num_labels=2)
    # from IPython import embed
    delta_model = AutoDeltaModel.from_config(config, model)
    delta_model.freeze_module(exclude = ['deltas','classifier'], set_state_dict = True)


    # delta_model.save_finetuned("autodelta_try", push_to_hub=True, private=True)
    delta_model = AutoDeltaModel.from_finetuned("ShengdingHu/autodelta_try", model, use_auth_token=True)




