import os
import re
from typing import Union, Dict, Any, Tuple, Optional
from  opendelta import __version__ as opendelta_version
from opendelta.utils import logging
from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
import transformers
import json
import copy

CONFIG_NAME = "config.json"
transformers_version = transformers.__version__

checked_package_versions = ["transformers_version", "opendelta_version"]

logger = logging.get_logger(__name__)
FULL_CONFIGURATION_FILE = "config.json"
_re_configuration_file = re.compile(r"config\.(.*)\.json")

class BaseDeltaConfig:
    r"""Base class for all configuration classes. Handles a few
    parameters common to all delta models' configurations as well as methods for loading/downloading/saving configurations.

    Class attributes (overridden by derived classes):

    - **delta_type** (:obj:`str`) -- the name of the delta modules, used to create the correct :py:class:`~opendelta.AutoConfig`.

    Args:
        modified_modules (:obj:`List[str]`, *optional*, defaults to :obj:`None`)
            The list of keys to determine which modules you want to modify. OpenDelta will take every modulees that
            **ends with** the one of the provided keys as the modification target. When not given any value, i.e.
            ``modified_modules=None``, the delta module will use the it corresponding default modification modules.
            Taking DistilBertModel with an classifier on top as an example:

            .. note::

                **Examples**: When adding delta to `DistilBertModel <https://huggingface.co/docs/transformers/model_doc/distilbert>`_,
                
                1. set to ``["0.attention.out_lin"]`` will add delta modules to the attention output of distilbert's layer 0, i.e., ``distilbert.transformer.layer.0.attention.out_lin``.
                2. set to ``["attention.out_lin"]`` will add the delta modules in every layer's ``attention.out_lin``.

        unfrozen_modules (:obj:`List[str]`, *optional*, defaults to :obj:`["deltas"]` ): The modules that are unfrozen
            during training in :meth:`~opendelta.basemodel.DeltaBase.freeze_module`, which includes the ones that are newly introduced as delta modules, and the ones that are originally a part of the model but set to trainable (:obj:`requires_grad=True`) to train together with the delta modules. Opendelta will take every modules that **ends with** the one of the provided keys and all its sub-modules and paramters as trainable.

        exclude_modules (:obj:`str`, *optional*, default to :obj:`None`): The modules starts with these strings will be excluded in modification. Note that currently only plain text (no regular expression) is supported.

            .. note::

                **Examples**: When adding delta to DistilBertModel,
        
                1. set this argument to ``["bias"]`` will make all bias terms tunable.
                2. set this argument to ``["attention"]`` will make all parameters in all attention modules tunable.
                3. set this argument to ``["deltas"]`` will make all the parameters in the newly introduced delta modules tunable.
                4. set this argument to ``["classifier"]`` will make all parameters in the classifier tunable.
                5. set this argument to ``["3.ffn.lin2", "deltas", "classifier"]``, will make all parameters in the third layer's feed forward layer's send linear layer, the detla modules, and the classifiers modules tunable.

        common_structure (:obj:`bool`, *optional*, default to :obj:`None`): Whether using the common structure mapping of the transformer model when designating ``modified_modules` and ``unfrozen_modules``.
        backbone_class (:obj:`str`, *optional*, default to :obj:`None`): The name of backbone model's class, e.g.
                ``RobertaForMaskedLM``. Saving this infomation let the users explicitly know on which backbone the
                delta model is trained.
        backbone_checkpoint_name (:obj:`str`, *optional*, default to :obj:`None`): The specific checkpoint of the model.
                In ideal case, it should be the url to download the checkpoint. However, we do not force the user to
                specify a downloadable url here.
        backbone_hash (:obj:`str`, *optional*, default to :obj:`None`): The md5-hash of the backbone model. It is
                calculated using the string representation of the model and the sequential expansion of all the
                parameters in the model. When loading a delta checkpoint in strict mode, the hash of the backbone model
                will be compared to the hash in this config.
    """
    delta_type: str = ""


    def __init__(self,
                 modified_modules = None,
                 exclude_modules = None,
                 unfrozen_modules = ["deltas"],
                 common_structure=False,
                 backbone_class = None,
                 backbone_checkpoint_name = None,
                 backbone_hash = None,
                 ):
        arg_names = get_arg_names(BaseDeltaConfig.__init__)
        for arg_name in arg_names:
            setattr(self, arg_name, locals()[arg_name])




    @classmethod
    def from_finetuned(cls, finetuned_delta_path: Union[str, os.PathLike], **kwargs) -> "BaseDeltaConfig":
        r"""
        Instantiate a :obj:`BaseDeltaConfig` (or a derived class) from a finetined delta module configuration.

        Args:
            finetuned_model_name_or_path (:obj:`str` or :obj:`os.PathLike`): This can be either:

                - a string, the *model id* of a finetuned delta model configuration hosted inside a model repo on
                  deltahub.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                  
                - a path to a *directory* containing a configuration file saved using the :meth:`BaseDeltaConfig.save_finetuned` method, e.g., ``./my_model_directory/``.
                
                - a path or url to a saved configuration JSON *file*, e.g., ``./my_model_directory/configuration.json``.

            cache_dir (:obj:`str` or :obj:`os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained delta model configuration should be cached if the
                standard cache should not be used.

        .. code-block:: python

            delta_config = AdapterConfig.from_finetuned("thunlp/FactQA_T5-large_Adapter", backbone_model=t5)

        """
        config_dict, kwargs = cls.get_config_dict(finetuned_delta_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warn(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    def save_finetuned(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a configuration object to the directory :obj:`save_directory`, so that it can be re-loaded using the
        :meth:`BaseDeltaConfig.from_finetuned` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`): Directory where the configuration JSON file
                will be saved (will be created if it does not exist).
            push_to_hub (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether or not to push your model to
                the Hugging Face model hub after saving it.

                .. warning::

                    1. Will raise error if you haven't config a Huggingface Model Hub.
                    2. Using ``push_to_hub=True`` will synchronize the repository you are pushing to with ``save_directory``, which requires ``save_directory`` to be a local clone of the repo you are pushing to if it's an existing folder. Pass along ``temp_dir=True`` to use a temporary directory instead.

            kwargs: Additional key word arguments.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")


    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "BaseDeltaConfig":
        r"""
        Instantiate a :obj:`BaseDeltaConfig` from a python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the :py:meth:`~PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.
        Returns:
            :obj:`BaseDeltaConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        accept_args = get_arg_names(cls.__init__) + get_arg_names(BaseDeltaConfig.__init__)
        unused_config_keys = []
        for config_key in list(config_dict.keys()):
            if config_key not in accept_args:
                config_dict.pop(config_key)
                unused_config_keys.append(config_key)
        logger.warning(f"The following keys are not used by {cls}.__init__ function: {unused_config_keys}")

        config = cls(**config_dict)


        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):

                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info(f"Model config\n{config}")

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def get_config_dict(
        cls, finetuned_delta_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """[NODOC]
        From a ``finetuned_delta_path``, resolve to a dictionary of parameters, to be used for instantiating a
        [``PretrainedConfig``] using ``from_dict``.
        Parameters:
            finetuned_delta_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.
        """
        cache_dir = kwargs.get("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        # resume_download = kwargs.pop("resume_download", False)
        # proxies = kwargs.pop("proxies", None)
        # use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.get("local_files_only", False)
        # revision = kwargs.pop("revision", None)
        # from_pipeline = kwargs.pop("_from_pipeline", None)
        # from_auto_class = kwargs.pop("_from_auto", False)

        # user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        # if from_pipeline is not None:
            # user_agent["using_pipeline"] = from_pipeline

        if os.environ.get("DELTACENTER_OFFLINE", '0') == '1':
            logger.info("Delta Center offline mode!")
            local_files_only = True

        finetuned_delta_path = str(finetuned_delta_path)

        if cache_dir is not None:
            cached_finetuned_delta_path = os.path.join(cache_dir, finetuned_delta_path)
        else:
            cached_finetuned_delta_path = finetuned_delta_path

        if os.path.isfile(cached_finetuned_delta_path):
            local_files_only = True
        elif os.path.isdir(cached_finetuned_delta_path):
            # cached_finetuned_delta_path = os.path.join(cached_finetuned_delta_path, 'config.json')
            local_files_only = True

        # if local_files_only:
        #     config_dict = cls._dict_from_json_file(cached_finetuned_delta_path)
        if not local_files_only or force_download:
            from .utils.delta_center import download as dcdownload
            # try to download from DeltaCenter
            cached_finetuned_delta_path = dcdownload(finetuned_delta_path, force_download=force_download, cache_dir=cache_dir)
            kwargs['force_download'] = False # Has been downloaded, not more forcing

        cached_finetuned_delta_path = os.path.join(cached_finetuned_delta_path, 'config.json')
        config_dict = cls._dict_from_json_file(cached_finetuned_delta_path)
        return config_dict, kwargs

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_json_string(self, use_diff: bool = True) -> str:
        """[NODOC]
        Serializes this instance to a JSON string.
        Args:
            use_diff (:obj:`bool`, *optional*, defaults to :obj:`True`):
                If set to :obj:`True`, only the difference between the config instance and the default ``PretrainedConfig()``
                is serialized to JSON string.
        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """[NODOC]
        Save this instance to a JSON file.
        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, *optional*, defaults to :obj:`True`):
                If set to :obj:`True`, only the difference between the config instance and the default ``PretrainedConfig()``
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def to_diff_dict(self) -> Dict[str, Any]:
        """[NODOC]
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = BaseDeltaConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() #if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                key not in default_config_dict
                or key in checked_package_versions
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        self.dict_torch_dtype_to_str(serializable_config_dict)

        return serializable_config_dict

    def update(self, config_dict: Dict[str, Any]):
        """[NODOC]
        Updates attributes of this class with attributes from ``config_dict``.
        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = transformers_version
        output["opendelta_version"] = opendelta_version

        self.dict_torch_dtype_to_str(output)

        return output

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """[NODOC]
        Checks whether the passed dictionary has a *torch_dtype* key and if it's not None, converts torch.dtype to a
        string of just the type. For example, ``torch.float32`` get converted into *"float32"* string, which can then be
        stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]




if __name__ == "__main__":
    myconfig = BaseDeltaConfig.from_pretrained("../ckpts/lora/")
    myconfig.save_pretrained("../ckpts/lora.1/")
    print(myconfig)