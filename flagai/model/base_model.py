# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from torch.nn import Module
import torch
import json
from typing import Union
from flagai.model.file_utils import _get_model_id, _get_checkpoint_path, _get_vocab_path, _get_model_files
import os


# The base model for models
class BaseModel(Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def forward(self, **kwargs):
        raise NotImplementedError("base model is not callable!")

    def load_weights(self, checkpoint_path):
        raise NotImplementedError("base model is not callable!")

    def save_weights(self, chekpoint_path=''):
        torch.save(self.state_dict, chekpoint_path)

    def save_config(self, config_path='./config.json'):
        with open(config_path, 'w', encoding='utf8') as jsonfile:
            json.dump(self.config, jsonfile, indent=4)

    @classmethod
    def init_from_json(cls, config_file='./config.json', **kwargs):
        with open(config_file, 'r', encoding='utf8') as js:
            args = json.load(js)
        for k in kwargs:
            args[k] = kwargs[k]
        if 'checkpoint_activations' not in args:
            args['checkpoint_activations'] = False
        return cls(args, **kwargs)

    @classmethod
    def _load_state_dict_into_model(cls,
                                    model,
                                    pretrained_model_name_or_path,
                                    verbose=False):
        pl_sd = torch.load(pretrained_model_name_or_path, map_location="cpu")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        m, u = model.load_state_dict(sd, strict=True)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        model.eval()
        return model

    @classmethod
    def from_pretrain(cls,
                      download_path='./checkpoints/',
                      model_name='RoBERTa-base-ch',
                      only_download_config=False,
                      device="cpu",
                      **kwargs):
        model_id = None

        raw_download_path = download_path
        # Try load model from local path
        download_path = os.path.join(download_path, model_name)

        config_path = os.path.join(download_path, "config.json")
        checkpoint_path = os.path.join(download_path, "pytorch_model.bin")

        def load_local(checkpoint_path):
            model = cls.init_from_json(config_path, **kwargs)
            model.to(device)
            if os.getenv('ENV_TYPE') != 'deepspeed+mpu':
                if os.path.exists(checkpoint_path):
                    model.load_weights(checkpoint_path)
            elif os.getenv('ENV_TYPE') == 'deepspeed+mpu':
                model_parallel_size = int(os.getenv("MODEL_PARALLEL_SIZE"))
                if torch.distributed.is_initialized(
                ) and torch.distributed.get_rank() == 0:
                    # change the mp_size in rank 0
                    print(
                        "preparing the model weights for model parallel size = {:02d}"
                        .format(model_parallel_size))
                    from flagai.auto_model.auto_loader import MODEL_DICT
                    from flagai.mp_tools import change_pytorch_model_mp_from_1_to_n_new, check_pytorch_model_mp_size
                    if model_parallel_size > 1 and not check_pytorch_model_mp_size(
                            download_path, model_parallel_size):
                        brief_model_name = MODEL_DICT[model_name.lower()][2]
                        change_pytorch_model_mp_from_1_to_n_new(
                            brief_model_name, download_path,
                            model_parallel_size)

                from flagai import mpu
                torch.distributed.barrier(group=mpu.get_model_parallel_group())

                if model_parallel_size > 1:
                    from flagai.mpu import get_model_parallel_rank
                    model_parallel_rank = get_model_parallel_rank()
                    checkpoint_path = os.path.join(
                        download_path,
                        "pytorch_model_{:02d}.bin".format(model_parallel_rank))
                    if os.path.exists(checkpoint_path):
                        model.load_weights(checkpoint_path)
                else:
                    model.load_weights(checkpoint_path)
            return model

        def load_diffusion_local(yaml_path, only_download_config=False, **kwargs):
            """
            Now only diffusion models requires yaml
            """
            from omegaconf import OmegaConf
            checkpoint_path = os.path.join(
                download_path, "model.ckpt")  # Specific to diffusion models

            config = OmegaConf.load(f"{yaml_path}")
            model_config = config.model
            model_config.params.cond_stage_config.params.download_path = raw_download_path
            kwargs.update(model_config.get("params", dict()))
            model = cls(**kwargs)
            if not only_download_config:
                model = cls._load_state_dict_into_model(
                    model,
                    checkpoint_path,
                )
            return model

        yaml_path = os.path.join(download_path, "config.yaml")
        if os.path.exists(yaml_path):
            """
            Now only diffusion models requires yaml
            """
            return load_diffusion_local(yaml_path, only_download_config=only_download_config, **kwargs)
        elif os.path.exists(config_path):
            """
            It is fine when checkpoint_path does not exist, for the case that only_download_config=True
            At that time the model will not be loaded. 
            """
            return load_local(checkpoint_path)

        try:
            model_id = _get_model_id(model_name)
        except:
            print("Model hub is not reachable!")
        # prepare the download path
        # downloading the files
        model: Union[Module, None]
        if model_id and model_id != "null":
            model_files = eval(_get_model_files(model_name))
            print("model files:" + str(model_files))
            for file_name in model_files:
                if not file_name.endswith("bin"):
                    _get_vocab_path(download_path, file_name, model_id)

            if not only_download_config and os.path.exists(
                    os.path.join(download_path, 'config.json')):
                if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
                    model_parallel_size = int(os.getenv("MODEL_PARALLEL_SIZE"))
                    if model_parallel_size > 1:
                        # if gpus == nums_of_modelhub_models
                        # can load
                        # else need to download the pytorch_model.bin and to recut.
                        model_hub_parallel_size = 0
                        for f in model_files:
                            if "pytorch_model_" in f:
                                model_hub_parallel_size += 1
                else:
                    model_parallel_size = 1

                if "pytorch_model_01.bin" in model_files and model_parallel_size > 1 and model_hub_parallel_size == model_parallel_size:
                    # Only to download the model slices(megatron-lm).
                    for file_to_load in model_files:
                        if "pytorch_model_" in file_to_load:
                            _get_checkpoint_path(download_path, file_to_load,
                                                 model_id)

                elif 'pytorch_model.bin' in model_files:
                    checkpoint_path = _get_checkpoint_path(
                        download_path, 'pytorch_model.bin', model_id)
                else:
                    checkpoint_merge = {}
                    # maybe multi weights files
                    for file_to_load in model_files:
                        if "pytorch_model-0" in file_to_load:
                            _get_checkpoint_path(download_path, file_to_load,
                                                 model_id)
                            checkpoint_to_load = torch.load(os.path.join(
                                download_path, file_to_load),
                                                            map_location="cpu")
                            for k, v in checkpoint_to_load.items():
                                checkpoint_merge[k] = v
                    # save all parameters
                    torch.save(
                        checkpoint_merge,
                        os.path.join(download_path, "pytorch_model.bin"))
        if os.path.exists(yaml_path):
            return load_diffusion_local(yaml_path,only_download_config=only_download_config)
        return load_local(checkpoint_path)

    @classmethod
    def download(cls,
                download_path='./checkpoints/',
                model_name='RoBERTa-base-ch',
                **kwargs):
        try:
            model_id = _get_model_id(model_name)
        except:
            print("Model hub is not reachable!")
        # prepare the download path
        # downloading the files
        if model_id and model_id != "null":
            model_files = eval(_get_model_files(model_name))
            print("model files:" + str(model_files))
            for file_name in model_files:
                if not file_name.endswith("bin"):
                    _get_vocab_path(os.path.join(download_path, model_name), file_name, model_id)
                else :
                    _get_checkpoint_path(os.path.join(download_path, model_name), file_name, model_id)