from .superglue import *

DATASET = {
    "BoolQ": BoolQ_Dataset,
    "CB": CB_Dataset,
    "COPA": COPA_Dataset,
    "MultiRC": MultiRC_Dataset,
    "ReCoRD": ReCoRD_Dataset,
    "RTE": RTE_Dataset,
    "WiC": WiC_Dataset,
    "WSC": WSC_Dataset,
}