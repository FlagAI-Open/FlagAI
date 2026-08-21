import torch 
from typing import List
import sys


def lora_transfer(model, env_args):
    from flagai.model.tools.peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    lora_config = LoraConfig(
        r=env_args.lora_r,
        lora_alpha=env_args.lora_alpha,
        target_modules=env_args.lora_target_modules,
        lora_dropout=env_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model
