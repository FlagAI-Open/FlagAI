
from typing import Dict, List, Union, Optional, Callable
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.model_md5 import gen_model_hash, gen_parameter_hash
import torch
import os
from opendelta import logging
import torch.nn as nn
from DeltaCenter import OssClient
import  yaml
from dataclasses import dataclass, field, fields
import datetime
from .file_utils import WEIGHTS_NAME

logger = logging.get_logger(__name__)



alternative_names = {
    "train_tasks":  ["train_tasks", "train_task", "task_name"],
}


@dataclass
class DeltaCenterArguments:
    """
    The arguments that are used to distinguish between different delta models on the DeltaCenter
    """
    name: str = field(default="",
                        metadata={"help": "The name of the delta model checkpoint"}
    )
    backbone_model: str = field(default="",
                                metadata={"help": "The backbone model of the delta model"}
    )
    backbone_model_path_public: str = field(
        default = None,
        metadata={"help": "Publicly available path (url) to pretrained model or model identifier from huggingface.co/models"}
    )
    delta_type: str = field(
        default=None,
        metadata={"help": "the type of type model, e.g., adapter, lora, etc."}
    )
    train_tasks: Optional[Union[List[str], str]]= field(
        default=None,
        metadata={"help": "the task(s） that the delta is trained on"}
    )
    train_datasets: Optional[Union[List[str], str]]= field(
        default=None,
        metadata={"help": "the datasets(s） that the delta is trained on"}
    )
    checkpoint_size: Optional[float] = field(
        default=None,
        metadata={"help": "the size of the checkpoint, in MB"}
    )
    test_tasks: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "the task(s) that the delta is tested on"}
    )
    test_datasets: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "the dataset(s) that the delta is tested on"}
    )
    test_performance: Optional[float] = field(
        default=None,
        metadata={"help": "the performance of the model on the test set"}
    )
    test_metrics: Optional[str] = field(
        default=None,
        metadata={"help": "the metrics used by the model"}
    )
    trainable_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "the ratio of trainable parameters in the model"}
    )
    delta_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "the ratio of delta parameters in the model"}
    )
    usage: Optional[str] = field(
        default="",
        metadata={"help": "the usage code of the model"}
    )
    license: Optional[str] = field(
        default="apache-2.0",
        metadata={"help": "the license of the model"}
    )



class SaveLoadMixin:
    def add_configs_when_saving(self,):
        self.config.backbone_class = self.backbone_model.__class__.__name__
        if hasattr(self.backbone_model, "config"):
            self.config.backbone_checkpoint_name = os.path.split(self.backbone_model.config._name_or_path.strip("/"))[-1]
        self.config.backbone_hash = gen_model_hash(self.backbone_model)



    def save_finetuned(
        self,
        finetuned_delta_path: Optional[Union[str, os.PathLike]] = "./delta_checkpoints/",
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_dc: bool = False,
        center_args: Optional[Union[DeltaCenterArguments, dict]] = dict(),
        center_args_pool: Optional[dict] = dict(),
        list_tags: Optional[List] = list(),
        dict_tags: Optional[Dict] = dict(),
        delay_push: bool = False,
        test_result = None,
        usage: Optional[str] = "",
    ):
        r"""
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        :py:meth:`~DeltaBase.save_finetuned` class method.

        Arguments:
            finetuned_delta_path: (optional) path to the directory where the model and its configuration file will be saved.
                If not specified, the model will be saved in the directory ``./delta_checkpoints/``,
                which is a subdirectory of the current working directory.
            save_config: (optional) if ``True``, the configuration file will be saved in the same directory as the
                model file. if ``False``, only the state dict will be saved.
            state_dict: (optional) a dictionary containing the model's state_dict. If not specified, the
                state_dict is loaded from the backbone model's trainable parameters.
            save_function: (optional) the function used to save the model. Defaults to ``torch.save``.
            state_dict_only: (optional) if ``True``, only the state_dict will be saved.
            push_to_dc: (optional) if ``True``, the model will prepare things to pushed to the DeltaCenter.
                This includes:
                - creating a configuration file for the model
                - creating a directory for the model
                - saving the model's trainable parameters
                - pushing the model to the DeltaCenter
            center_args: (optional) the arguments that are used to distinguish between different delta models on the DeltaCenter
            center_args_pool: (optional) a dictionary containing the arguments that are used to distinguish between different delta models on the DeltaCenter
            list_tags: (optional) a list of tags that will be added to the model's configuration file
            dict_tags: (optional) a dictionary of tags that will be added to the model's configuration file
            delay_push: (optional) if ``True``, the model will not be pushed to the DeltaCenter. This is useful if you want to
                push the model later.

        """

        # create the config to save, including model hash, etc.
        if save_config:
            if not hasattr(self, "config"):
                self.create_config_from_model()
            self.add_configs_when_saving()

        if push_to_dc:
            final_center_args = self.create_delta_center_args(center_args=center_args,
                        center_args_pool=center_args_pool)

        save_directory = finetuned_delta_path
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_dc:
            save_directory = os.path.join(save_directory, final_center_args.name)
            os.makedirs(save_directory, exist_ok=True)

        model_to_save = self.backbone_model# unwrap_model(self)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

         # Save the config
        if save_config:
            self.config.save_finetuned(save_directory)
        


        


        logger.info("\n"+"*"*30+f"\nYou delta models has been saved locally to:\t{os.path.abspath(save_directory)}"
                 )
        self.compute_saving(output_model_file)

        state_dict_total_params = sum(p.numel() for p in state_dict.values())
        other_tags={}
        other_tags.update({'state_dict_total_params(M)':state_dict_total_params/1024/1024})
        other_tags.update({'test_result':test_result})
        if push_to_dc:
            logger.info("Creating yaml file for delta center")
            self.create_yml(save_directory, final_center_args, list_tags, dict_tags, other_tags)

            if not delay_push:
                OssClient.upload(base_dir=save_directory)
            else:
                logger.info(f"Delay push: you can push it to the delta center later using \n\tpython -m DeltaCenter upload {os.path.abspath(save_directory)}\n"
                    +"*"*30)
        else:
            logger.info("We encourage users to push their final and public models to delta center to share them with the community!")

    def compute_saving(self, output_model_file):
        import os
        stats = os.stat(output_model_file)
        if stats.st_size > (1024**3):
            unit = 'GB'
            value = stats.st_size/(1024**3)
        else:
            unit = 'MB'
            value = stats.st_size/(1024**2)
        logger.info("The state dict size is {:.3f} {}".format(value, unit))




    def create_yml(self, save_dir, config, list_tags=list(), dict_tags=dict(),other_tags=None):
        f = open("{}/config.yml".format(save_dir), 'w')
        config_dict = vars(config)
        config_dict['dict_tags'] = dict_tags
        config_dict['list_tags'] = list_tags
        if other_tags is not None:
            config_dict.update(other_tags)
        yaml.safe_dump(config_dict, f)
        f.close()

    def load_checkpoint(self, path, load_func=torch.load, backbone_model=None):
        r"""Simple method for loading only the checkpoint
        """
        if backbone_model is None:
            backbone_model = self.backbone_model
        self.backbone_model.load_state_dict(load_func(f"{path}/{WEIGHTS_NAME}"), strict=False)

    def save_checkpoint(self, path, save_func=torch.save, backbone_model=None):
        r"""Simple method for saving only the checkpoint"""
        if backbone_model is None:
            backbone_model = self.backbone_model
        save_func(backbone_model.state_dict(), f"{path}/{WEIGHTS_NAME}")

    @classmethod
    def from_finetuned(cls,
                       finetuned_delta_path: Optional[Union[str, os.PathLike]],
                       backbone_model: nn.Module,
                       delta_config = None,
                       cache_dir: Optional[Union[str, os.PathLike]] = None,
                       state_dict: Optional[dict] = None,
                       *model_args,
                       force_download: Optional[bool] = False,
                       check_hash: Optional[bool] = True,
                       local_files_only: Optional[bool] = False,
                       **kwargs):
        r"""
        Instantiate a finetuned delta model from a path.
        The backbone_model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated).
        To further train the model, you can use the :meth:`freeze_module <opendelta.basemodel.DeltaBase.freeze_module>` method.

        Parameters:
            finetuned_delta_path: (optional) path to the directory where the model and its configuration file will be saved.
                If not specified, the model will be loaded from the directory cahce directory. (see ``cache_dir``),
            backbone_model:  the backbone model that will be used to instantiate the finetuned delta model.
            delta_config: (optional) the configuration file of the finetuned delta model. If not specified, the configuration file
                is loaded from the directory ``finetuned_delta_path``.
            cache_dir: (optional) path to the directory where the model and its configuration file will be saved.
                If not specified, we will first look into current working directory, then the cache directory of your system, e.g., ~/.cache/delta_center/,
            state_dict: (optional) a dictionary containing the model's state_dict. If not specified, the
                state_dict is loaded from the ``finetuned_delta_path``.
            force_download: (optional) if ``True``, the model will be downloaded from the internet even if it is already
                present in the cache directory.
            check_hash: (optional) if ``True``, check whether the hash of the model once it's trained differs from what we load now.
            local_files_only: (optional) if ``True``, the model will be loaded from the local cache directory.
        """

        if os.environ.get("DELTACENTER_OFFLINE", '0') == '1':
            logger.info("Delta Center offline mode!")
            local_files_only = True

        # Load config if we don't provide a configuration


        finetuned_delta_path = str(finetuned_delta_path)

        if cache_dir is not None:
            cached_finetuned_delta_path = os.path.join(cache_dir, finetuned_delta_path)
        else:
            cached_finetuned_delta_path = finetuned_delta_path

        download_from_dc = False
        if os.path.isfile(cached_finetuned_delta_path):
            raise RuntimeError(
                        f"You should pass a directory to load a delta checkpoint instead of a file, "
                        f"since we need the delta's configuration file."
                    )
        elif os.path.isdir(cached_finetuned_delta_path):
            if os.path.isfile(os.path.join(cached_finetuned_delta_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                weight_file = os.path.join(cached_finetuned_delta_path, WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    f"Error no file named {WEIGHTS_NAME} found in "
                    f"directory {cached_finetuned_delta_path}."
                )

        else:
            # try to download from DeltaCenter
            from .delta_center import download as dcdownload
            cached_finetuned_delta_path = dcdownload(finetuned_delta_path, cache_dir=cache_dir, force_download=force_download)
            download_from_dc = True
            weight_file = os.path.join(cached_finetuned_delta_path, WEIGHTS_NAME)

        if state_dict is None:
            state_dict = torch.load(weight_file, map_location="cpu")

        if not isinstance(delta_config, BaseDeltaConfig):
            delta_config, model_kwargs = cls.config_class.from_finetuned(
                cached_finetuned_delta_path,
                cache_dir=None,
                return_unused_kwargs=True,
                local_files_only=True if download_from_dc else local_files_only, # has been downloaded
                **kwargs,
            )

        else:
            model_kwargs = kwargs


        # Initialize the model from config and attach the delta model to the backbone_model.
        delta_model = cls.from_config(delta_config, backbone_model, *model_args, **model_kwargs, )

        # load the state_dict into the backbone_model. As the delta model's parameter
        # is the same object as the deltas in the backbone model with different reference name,
        # the state_dict will also be loaded into the delta model.
        delta_model._load_state_dict_into_backbone(backbone_model, state_dict)

        backbone_hash = gen_model_hash(backbone_model)

        if check_hash:
            if hasattr(delta_config, "backbone_hash") and \
                    delta_config.backbone_hash is not None and \
                    delta_config.backbone_hash != backbone_hash:
                logger.warning("The config has an hash of the backbone model, and is"
                                "different from the hash of the loaded model. This indicates a mismatch"
                                "between the backbone model that the delta checkpoint is based on and"
                                "the one you loaded. You propobability need to Train the model instead of"
                                "directly inference. ")
            else:
                logger.info("Hash-check passed. You can safely use this checkpoint directly.")
        else:
            logger.warning("Parameters' hash has not been checked!")


        # Set model in evaluation mode to deactivate DropOut modules by default
        backbone_model.eval()

        return delta_model


    def create_delta_center_args(self, center_args, center_args_pool):
        """
        Create the delta center args for the center model.
        center_args has higher priority than center_args_pool.

        """
        mdict = {}
        field = fields(DeltaCenterArguments)


        for f in field:
            exist = False
            # first is center_args, exact match
            if f.name in center_args:
                mdict[f.name] = center_args[f.name]
                continue
            # second is center_args_pool, can use alternative names
            if f.name in center_args_pool:
                mdict[f.name] = center_args_pool[f.name]
                exist = True
            elif f.name in alternative_names:
                for altername in alternative_names[f.name]:
                    if altername in center_args_pool:
                        mdict[f.name] = center_args_pool[altername]
                        exist = True
                        break
            # if not exist, find from self.stat or set to default
            if not exist:
                if f.name in self.stat:
                    mdict[f.name] = self.stat[f.name]
                else:
                    mdict[f.name] = f.default

        # if eventualy name is not set, create a default one
        if mdict['name'] is None or mdict['name'] == '':
            logger.info("Name is not set, use default name.")
            mdict['name'] = self.create_default_name(**mdict)

        if len(mdict['usage']) == 0:
            logger.info("Usage is not set, use default usage.")
            mdict['usage'] = self.create_default_usage(mdict['name'])


        center_args = DeltaCenterArguments(**mdict)
        return  center_args

    def create_default_usage(self, name):
        usage_str = """from opendelta import AutoDeltaModel\n""" + \
            """delta_model = AutoDeltaModel.from_finetuned('{name_with_userid}', backbone_model=model)\n""" + \
            """delta_model.freeze_module() # if you are going to further train it \n""" + \
            """delta_model.log()"""
        return usage_str

    def create_default_name(self, **kwargs):
        r"""Currently, it's only a simple concatenation of the arguments.
        """

        reponame = ""
        reponame += kwargs["backbone_model_path_public"].split("/")[-1]+"_" if kwargs['backbone_model_path_public'] is not None else kwargs['backbone_model']
        reponame += kwargs["delta_type"]+"_" if kwargs["delta_type"] is not None else ""

        # tasks
        if isinstance(kwargs["train_tasks"], list):
            train_tasks = "+".join(kwargs["train_tasks"])
        elif kwargs["train_tasks"] is not None:
            train_tasks = kwargs["train_tasks"]
        else:
            logger.warning("train_tasks are not find in all arguments. Do you miss them?")
            train_tasks = None
        reponame += train_tasks+"_" if train_tasks is not None else ""

        # time
        reponame += datetime.datetime.now().strftime("%Y%m%d%H%M%S") #+ gen_model_hash(model=self.backbone_model)

        # model hash
        if hasattr(self.config, "backbone_hash"):
            reponame += self.config.backbone_hash[:3]
        return reponame

