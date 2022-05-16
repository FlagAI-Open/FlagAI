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
                      model_name='RoBERTa-wwm-ext',
                      only_download_config=False,
                      **kwargs):
        model_id = _get_model_id(model_name)
        config_path = None
        download_path = os.path.join(download_path, model_name)
        checkpoint_path = os.path.join(download_path, "pytorch_model.bin")
        # prepare the download path
        # downloading the files
        model: Union[Module, None]
        if model_id != "null":
            _get_vocab_path(download_path, "vocab.txt", model_id)
            if not only_download_config:
                checkpoint_path = _get_checkpoint_path(download_path,
                                                       'pytorch_model.bin',
                                                       model_id)

        config_path = os.path.join(download_path, "config.json")
        if not os.path.exists(config_path) and model_id != "null":
            config_path = _get_config_path(download_path, 'config.json',
                                           model_id)
        if os.path.exists(config_path):
            model = cls.init_from_json(config_path, **kwargs)
            if os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
        else:
            model = None
        return model
