# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from sklearn.linear_model import HuberRegressor
from torch.nn import Module
import torch
import json
from typing import Union 
from flagai.model.file_utils import _get_model_id, _get_config_path, _get_checkpoint_path, _get_vocab_path, _get_model_files
import os
from glob import glob

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
    def from_pretrain(cls,
                      download_path='./checkpoints/',
                      model_name='RoBERTa-base-ch',
                      only_download_config=False,
                      device="cpu",
                      **kwargs):
        model_id = None
        try:
            model_id = _get_model_id(model_name)
        except:
            print("Model hub is not reachable!")
        config_path = None
        download_path = os.path.join(download_path, model_name)
        checkpoint_path = os.path.join(download_path, "pytorch_model.bin")
        # prepare the download path
        # downloading the files
        model: Union[Module, None]
        if model_id and model_id != "null":
            model_files = eval(_get_model_files(model_name))
            if not os.path.exists(os.path.join(download_path, 'vocab.txt')):
                if "vocab.txt" in model_files:
                    _get_vocab_path(download_path, "vocab.txt", model_id)

            if not only_download_config and not os.path.exists(os.path.join(download_path, 'config.json')):
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
                            _get_checkpoint_path(download_path,
                                                 file_to_load,
                                                 model_id)

                elif 'pytorch_model.bin' in model_files:
                    checkpoint_path = _get_checkpoint_path(download_path,
                                                           'pytorch_model.bin',
                                                           model_id)
                else :
                    checkpoint_merge = {}
                    # maybe multi weights files
                    for file_to_load in model_files:
                        if "pytorch_model-0" in file_to_load:
                            _get_checkpoint_path(download_path,
                                                 file_to_load,
                                                 model_id)
                            checkpoint_to_load = torch.load(os.path.join(download_path, file_to_load), map_location="cpu")
                            for k, v in checkpoint_to_load.items():
                                checkpoint_merge[k] = v
                    # save all parameters
                    torch.save(checkpoint_merge, os.path.join(download_path, "pytorch_model.bin"))

        config_path = os.path.join(download_path, "config.json")
        if model_id and not os.path.exists(config_path) and model_id != "null":
            config_path = _get_config_path(download_path, 'config.json',
                                           model_id)
        if os.path.exists(config_path):
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
                        change_pytorch_model_mp_from_1_to_n_new(brief_model_name,
                            download_path, model_parallel_size)

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
        else:
            model = None
        return model
