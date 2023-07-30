# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from torch.nn import Module
import torch
import json
from typing import Union
from flagai.model.file_utils import _get_model_id, _get_checkpoint_path, _get_vocab_path, _get_model_files
import os
class ConfigObj:

    def __init__(self):
        self.json_config = None

    def  __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)
    
        
def change_json_to_cls(jsonobj):
    obj = ConfigObj()
    for key in jsonobj:
        setattr(obj, key, jsonobj[key])

    obj.json_config = jsonobj
    
    return obj
    
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
            if isinstance(self.config, ConfigObj):
                config = self.config.json_config
            else :
                config = self.config 
            json.dump(config, jsonfile, indent=4)

    @classmethod
    def init_from_json(cls, config_file='./config.json', device='cpu', **kwargs):
        with open(config_file, 'r', encoding='utf8') as js:
            args = json.load(js)
        for k in kwargs:
            args[k] = kwargs[k]
        if 'checkpoint_activations' not in args:
            args['checkpoint_activations'] = False
        if 'use_cache' not in args:
            args['use_cache'] = False
        if "fp16" in kwargs and kwargs["fp16"] == True:
            if device == "cpu":
                torch.set_default_tensor_type(torch.HalfTensor)
            else:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
            model = cls(change_json_to_cls(args), **kwargs)
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            model = cls(change_json_to_cls(args), **kwargs)
        return model

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

        def load_local(checkpoint_path, only_download_config=False):
            model = cls.init_from_json(config_path, device=device, **kwargs)
            model.to(device)
            if only_download_config:
                return model 
            if 'adapter_dir' in kwargs:
                from flagai.model.tools.peft import PeftModel
                model = PeftModel.from_pretrained(model, kwargs['adapter_dir'])
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
            return load_local(checkpoint_path, only_download_config=only_download_config)

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
        return load_local(checkpoint_path, only_download_config=only_download_config)

    @classmethod
    def download(cls,
                download_path='./checkpoints/',
                model_name='RoBERTa-base-ch',
                only_download_config=False,
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
                if not file_name.endswith("bin") and not file_name.endswith("pth"):
                    _get_vocab_path(os.path.join(download_path, model_name), file_name, model_id)
                else :
                    if only_download_config:
                        continue
                    _get_checkpoint_path(os.path.join(download_path, model_name), file_name, model_id)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
