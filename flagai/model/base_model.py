from sklearn.linear_model import HuberRegressor
from torch.nn import Module
import torch
import json
from typing import Union 
from flagai.model.file_utils import _get_model_id, _get_config_path, _get_checkpoint_path, _get_vocab_path
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
        return cls(args, **kwargs)

    @classmethod
    def from_pretrain(cls,
                      download_path='./checkpoints/',
                      model_name='RoBERT-base-ch',
                      only_download_config=False,
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
            if not os.path.exists(os.path.join(download_path, 'vocab.txt')):
                _get_vocab_path(download_path, "vocab.txt", model_id)
            if not only_download_config and not os.path.exists(os.path.join(download_path, 'config.json')):
                checkpoint_path = _get_checkpoint_path(download_path,
                                                       'pytorch_model.bin',
                                                       model_id)

        config_path = os.path.join(download_path, "config.json")
        if model_id and not os.path.exists(config_path) and model_id != "null":
            config_path = _get_config_path(download_path, 'config.json',
                                           model_id)
        if os.path.exists(config_path):
            model = cls.init_from_json(config_path, **kwargs)
            if os.getenv('ENV_TYPE')!='deepspeed+mpu':
                if os.path.exists(checkpoint_path):
                    model.load_weights(checkpoint_path)
            elif os.getenv('ENV_TYPE')=='deepspeed+mpu':
                model_parallel_size = int(os.getenv("MODEL_PARALLEL_SIZE"))
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    # change the mp_size in rank 0
                    print("preparing the model weights for model parallel size = {:02d}".format(model_parallel_size))
                    from flagai.mp_tools import change_pytorch_model_mp_from_1_to_n, check_pytorch_model_mp_size
                    if model_parallel_size>1 and not check_pytorch_model_mp_size(download_path, model_parallel_size):
                        change_pytorch_model_mp_from_1_to_n(download_path, model_parallel_size)
                if model_parallel_size>1:
                    from flagai.mpu import get_model_parallel_rank
                    model_parallel_rank = get_model_parallel_rank()
                    checkpoint_path = os.path.join(download_path,
                                            "pytorch_model_{:02d}.bin".format(model_parallel_rank))
                    if os.path.exists(checkpoint_path):
                        model.load_weights(checkpoint_path)
                else:
                    model.load_weights(checkpoint_path) 
        else:
            model = None
        return model
