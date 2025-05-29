# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import copy
from typing import Any, Dict, Union
from ...utils import check_web_and_convert_path

class Config(object):
    """ enc_dec model configuration """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **args):
        path = check_web_and_convert_path(pretrained_model_name_or_path, 'config')
        return cls.from_json_file(os.path.join(path, 'config.json'), **args)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **args):
        config_dict = cls._dict_from_json_file(json_file, **args)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike], **args):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        res = json.loads(text)
        for key in args:
            res[key] = args[key]
        return res

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