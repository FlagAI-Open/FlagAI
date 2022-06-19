# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
try:
    from apex.optimizers import FusedAdam as Adam
except:
    from torch.optim import Adam
from .fp16 import FP16_Optimizer


def get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}

    # check & avoid shared weights
    def ids_list(group):
        return [id(param) for param in group]

    def check_param(param, group):
        param_id = id(param)
        return param_id in ids_list(group['params'])

    for module_ in module.modules():
        #if isinstance(module_, (nets.LayerNorm, torch.nn.LayerNorm)):
        if 'norm' in module_._get_name().lower():
            no_weight_decay_params['params'].extend([
                p for p in list(module_._parameters.values())
                if p is not None and p.requires_grad
            ])
        else:
            wdw = [
                p for n, p in list(module_._parameters.items())
                if p is not None and p.requires_grad and n != 'bias'
            ]
            no_wdw = [
                p for n, p in list(module_._parameters.items())
                if p is not None and p.requires_grad and n == 'bias'
            ]
            wdw = [p for p in wdw if not check_param(p, weight_decay_params)]
            no_wdw = [
                p for p in no_wdw if not check_param(p, no_weight_decay_params)
            ]
            weight_decay_params['params'].extend(wdw)
            no_weight_decay_params['params'].extend(no_wdw)

    return weight_decay_params, no_weight_decay_params


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    while hasattr(model, 'module'):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        # print('## param_group', len(param_group['params']))
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    return param_groups


def get_optimizer(param_groups,
                  lr=1e-3,
                  weight_decay=0,
                  adam_beta1=0.9,
                  adam_beta2=0.999,
                  adam_eps=1e-8,
                  cpu_optimizer=False,
                  cpu_torch_adam=False,
                  optimizer='adam',
                  fp16=False,
                  loss_scale=0,
                  dynamic_loss_scale=True,
                  loss_scale_window=1000,
                  min_scale=1,
                  hysteresis=2):
    """Set up the optimizer."""
    if cpu_optimizer:
        # Apex FusedAdam uses decoupled weight decay so use the same here
        if cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=lr,
                                       weight_decay=weight_decay)
    else:
        # Use FusedAdam.
        if optimizer == 'adam':
            optimizer = Adam(param_groups,
                             lr=lr,
                             weight_decay=weight_decay,
                             betas=(adam_beta1, adam_beta2),
                             eps=adam_eps)
        elif optimizer == 'adafactor':
            from transformers import Adafactor
            optimizer = Adafactor(param_groups,
                                  lr=lr,
                                  relative_step=False,
                                  warmup_init=False)
        else:
            raise NotImplementedError

    print(f'Optimizer = {optimizer.__class__.__name__}')
    # Wrap into fp16 optimizer.
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': loss_scale_window,
                                       'min_scale': min_scale,
                                       'delayed_shift': hysteresis
                                   })

    return optimizer
