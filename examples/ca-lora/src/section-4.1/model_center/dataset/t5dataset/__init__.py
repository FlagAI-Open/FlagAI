from .superglue import *
from .glue import *
from .squad import *
from .cnndm import *

DATASET = {
    "BoolQ": BoolQ_Dataset,
    "CB": CB_Dataset,
    "COPA": COPA_Dataset,
    "MultiRC": MultiRC_Dataset,
    "ReCoRD": ReCoRD_Dataset,
    "RTE": RTE_Dataset,
    "WiC": WiC_Dataset,
    "WSC": WSC_Dataset,
    "COPA_MLM": COPA_MLM_Dataset,
    "RTE_MLM": RTE_MLM_Dataset,
    "MNLI": MNLI_Dataset,
    "QQP": QQP_Dataset,
    "QNLI": QNLI_Dataset,
    "SST2": SST2_Dataset,
    "MRPC": MRPC_Dataset,
    "SQuAD": SQuAD_Dataset,
    "CNNDM": CNNDM_Dataset,
}