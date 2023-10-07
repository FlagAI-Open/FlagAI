
import os
from flagai.model.aquila2.conversation import Conversation, get_conv_template
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)
from typing import Dict, List, Optional
from flagai.model.aquila2.llama_condense_monkey_patch import (
    replace_llama_with_condense,
)

class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")


def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

# A global registry for all model adapters
# TODO (lmzheng): make it a priority queue.
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())
    

class AquilaChatAdapter(BaseModelAdapter):
    """The model adapter for BAAI/AquilaChat-7B"""

    def match(self, model_path: str):
        return "aquila" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")

        ## Long Context
        config = AutoConfig.from_pretrained(model_path, revision=revision)
        if config.rope_scaling["factor"] is not None:
            replace_llama_with_condense(config.rope_scaling["factor"])

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        model = model.train()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "aquila-cpt-v2" in model_path:
            return get_conv_template("aquila-cpt-v2")
        elif "aquila-cpt-v1" in model_path:
            return get_conv_template("aquila-cpt-v1")
        elif "aquila-v1" in model_path:
            return get_conv_template("aquila-v1")
        elif "aquila-v2" in model_path:
            return get_conv_template("aquila-v2")
        elif "aquila-v3" in model_path:
            return get_conv_template("aquila-v3")
        elif "aquila-v4" in model_path:
            return get_conv_template("aquila-v4")
        elif "aquila-chat" in model_path:
            return get_conv_template("aquila-chat")
        else:
            return get_conv_template("aquila")
        
# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
# register_model_adapter(PeftModelAdapter)
# register_model_adapter(StableLMAdapter)
# register_model_adapter(BardAdapter)
# register_model_adapter(ChatGPTAdapter)
# register_model_adapter(ClaudeAdapter)
# register_model_adapter(FalconAdapter)
# register_model_adapter(Llama2Adapter)
# register_model_adapter(QwenChatAdapter)
register_model_adapter(AquilaChatAdapter)
# register_model_adapter(Lamma2ChineseAdapter)
# register_model_adapter(OpenLLaMaOpenInstructAdapter)


# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)
