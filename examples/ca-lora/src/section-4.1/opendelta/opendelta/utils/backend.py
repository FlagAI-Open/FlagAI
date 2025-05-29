

import importlib


class BackendMapping:
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:

        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, backend):
        self.backend = backend
        assert backend in ['hf', 'bmt'], "Backend should be one of 'hf', 'bmt'. "
        if backend == 'hf':
            self.backend_mapping = {
                    "linear": "torch.nn.Linear",
                    "layer_norm": "torch.nn.LayerNorm",
                    "module": "torch.nn.Module",
                    "parameter": "torch.nn.Parameter"
                }
        elif backend == 'bmt':
            self.backend_mapping = {
                    "linear": "model_center.layer.Linear",
                    "layer_norm": "model_center.layer.LayerNorm",
                    "module": "bmtrain.layer.DistributedModule",
                    "parameter": "bmtrain.nn.DistributedParameter"
                }
        self.registered = {}

    def load(self, model_type):
        if model_type not in self.registered:
            splited = self.backend_mapping[model_type].split(".")
            module_name, class_name  = ".".join(splited[:-1]), splited[-1]
            module = importlib.import_module(module_name)
            the_class =  getattr(module, class_name)
            self.registered[model_type] = the_class
        return self.registered[model_type]

    def check_type(self, module, expect_type):
        the_class = self.load(expect_type)
        if isinstance(module, the_class):
            return True
        else:
            return False


    # def keys(self):
    #     mapping_keys = [
    #         self._load_attr_from_module(key, name)
    #         for key, name in self._config_mapping.items()
    #         if key in self._model_mapping.keys()
    #     ]
    #     return mapping_keys + list(self._extra_content.keys())

    # def get(self, key, default):
    #     try:
    #         return self.__getitem__(key)
    #     except KeyError:
    #         return default

    # def __bool__(self):
    #     return bool(self.keys())

    # def values(self):
    #     mapping_values = [
    #         self._load_attr_from_module(key, name)
    #         for key, name in self._model_mapping.items()
    #         if key in self._config_mapping.keys()
    #     ]
    #     return mapping_values + list(self._extra_content.values())

    # def items(self):
    #     mapping_items = [
    #         (
    #             self._load_attr_from_module(key, self._config_mapping[key]),
    #             self._load_attr_from_module(key, self._model_mapping[key]),
    #         )
    #         for key in self._model_mapping.keys()
    #         if key in self._config_mapping.keys()
    #     ]
    #     return mapping_items + list(self._extra_content.items())

    # def __iter__(self):
    #     return iter(self.keys())

    # def __contains__(self, item):
    #     if item in self._extra_content:
    #         return True
    #     if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
    #         return False
    #     model_type = self._reverse_config_mapping[item.__name__]
    #     return model_type in self._model_mapping

    # def register(self, key, value):
    #     """
    #     Register a new model in this mapping.
    #     """
    #     if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
    #         model_type = self._reverse_config_mapping[key.__name__]
    #         if model_type in self._model_mapping.keys():
    #             raise ValueError(f"'{key}' is already used by a Transformers model.")

    #     self._extra_content[key] = value


