
__version__ = "0.3.0"

class GlobalSetting:
    def __init__(self):
        self.axis_order = [0,1,2]


global_setting = GlobalSetting()

from .delta_configs import BaseDeltaConfig
from .utils import logging
from .utils.saving_loading_utils import SaveLoadMixin
from .basemodel import DeltaBase
from .auto_delta import AutoDeltaConfig, AutoDeltaModel
from .utils.structure_mapping import CommonStructureMap
from .delta_models.lora import LoraModel
from .delta_models.bitfit import BitFitModel
from .delta_models.compacter import CompacterModel
from .delta_models.adapter import AdapterModel
from .delta_models.prefix import PrefixModel
from .delta_models.soft_prompt import SoftPromptModel
from .delta_models.low_rank_adapter import LowRankAdapterModel
from .delta_models.parallel_adapter import ParallelAdapterModel



def set_axis_order(axis_order=[0,1,2]):
    setattr(global_setting, "axis_order", axis_order)