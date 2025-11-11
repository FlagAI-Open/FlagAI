# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
""" enc_dec model configuration """

import json
import os
import copy
from typing import Any, Dict, Tuple, Union
import torch


class Config(object):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: Union[str,
                                                             os.PathLike]):
        return cls.from_json_file(pretrained_model_name_or_path)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output
