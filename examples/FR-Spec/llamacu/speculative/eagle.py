from .. import C
from .tree_drafter import LLM_with_tree_drafter

import torch
from transformers import PretrainedConfig

class EagleConfig(PretrainedConfig):
    def __init__(
        self,
        num_hidden_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eagle_num_layers = num_hidden_layers

class LLM_with_eagle(LLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 V=15000,
                 **kwargs):
        super().__init__(
            "eagle", eagle_path, base_path,
            tree_size = tree_size,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = EagleConfig.from_pretrained(eagle_path)

        C.init_eagle_model(
            self.eagle_config.eagle_num_layers,
            num_iter,
            topk_per_iter,
            self.tree_size,
            V,
            self.dtype_int,
        )

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if name == "token_id_remap":
                C.load_model(f"{cls}.{name}", param.data_ptr())
                return
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous().to(dtype)
            if 'embed_tokens' in name:
                return
            if 'fc' in name:
                if 'weight' in name:
                    param1 = param[..., :param.shape[-1] // 2].contiguous()
                    param2 = param[..., param.shape[-1] // 2:].contiguous()
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param1.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc2')}", param2.data_ptr())
                else: # bias
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param.data_ptr())
            else:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)
