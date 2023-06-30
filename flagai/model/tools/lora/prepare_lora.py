import torch 
from typing import List
import sys

def lora_transfer(model, env_args):
    # from flagai.model.tools.lora import (
    #     LoraConfig,
    #     get_peft_model,
    #     get_peft_model_state_dict,
    #     prepare_model_for_int8_training,
    #     set_peft_model_state_dict,
    # )
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    # Added for Lora
    
    lora_config = LoraConfig(
        r=env_args.lora_r,
        lora_alpha=env_args.lora_alpha,
        target_modules=env_args.lora_target_modules,
        lora_dropout=env_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 2. Prepare model
    # model = prepare_model_for_kbit_training(model)
    # model.tok_embeddings.weight.requires_grad = True
    # for name, param in model.named_parameters():
    #     # all_param += param.numel()
    #     if param.requires_grad:
    #         print(name)
    model = get_peft_model(model, lora_config)

    # model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    return model
