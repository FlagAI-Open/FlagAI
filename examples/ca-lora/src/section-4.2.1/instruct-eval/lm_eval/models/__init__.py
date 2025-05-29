from . import gpt2
from . import gpt3
from . import textsynth
from . import dummy
from . import llama
from . import qlora_CA-LoRA

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "llama": llama.LlamaLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
    "qlora_CA-LoRA": qlora_CA-LoRA.QloraCA-LoRALM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
