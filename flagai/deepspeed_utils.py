# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
DeepSpeed API 兼容性适配模块
用于适配 DeepSpeed 0.18.2 与旧版本之间的 API 差异
"""

import packaging.version
import warnings

try:
    import deepspeed
    HAS_DEEPSPEED = True
    DS_VERSION = packaging.version.parse(deepspeed.__version__)
except ImportError:
    HAS_DEEPSPEED = False
    DS_VERSION = None


def get_deepspeed_version():
    """获取 DeepSpeed 版本"""
    if not HAS_DEEPSPEED:
        return None
    return DS_VERSION


def is_deepspeed_version(version_str):
    """检查 DeepSpeed 版本是否满足要求"""
    if not HAS_DEEPSPEED:
        return False
    required_version = packaging.version.parse(version_str)
    return DS_VERSION >= required_version


def initialize_deepspeed(model, model_parameters=None, optimizer=None, 
                         lr_scheduler=None, mpu=None, config=None,
                         dist_init_required=None, **kwargs):
    """
    兼容新旧版本的 DeepSpeed 初始化
    
    Args:
        model: 模型
        model_parameters: 模型参数
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        mpu: 模型并行工具
        config: DeepSpeed 配置
        dist_init_required: 是否需要初始化分布式（新版本可能已废弃）
        **kwargs: 其他参数
    
    Returns:
        DeepSpeed 引擎和相关对象
    """
    if not HAS_DEEPSPEED:
        raise ImportError("DeepSpeed is not installed. Please install it with: pip install deepspeed>=0.18.2")
    
    # DeepSpeed 0.9.0+ 版本
    if is_deepspeed_version("0.9.0"):
        init_kwargs = {
            "model": model,
            "config": config or {},
        }
        
        # 添加可选参数
        if model_parameters is not None:
            init_kwargs["model_parameters"] = model_parameters
        if optimizer is not None:
            init_kwargs["optimizer"] = optimizer
        if lr_scheduler is not None:
            init_kwargs["lr_scheduler"] = lr_scheduler
        
        # MPU 参数处理（新版本可能通过 config 配置）
        # 如果提供了 mpu，可能需要特殊处理
        if mpu is not None:
            # 新版本可能不再需要 mpu 参数，或者通过其他方式配置
            # 检查是否有 mpu 参数
            import inspect
            sig = inspect.signature(deepspeed.initialize)
            if 'mpu' in sig.parameters:
                init_kwargs["mpu"] = mpu
        
        # dist_init_required 在新版本中可能已废弃
        # 只在旧版本中使用
        if dist_init_required is not None and not is_deepspeed_version("0.9.0"):
            init_kwargs["dist_init_required"] = dist_init_required
        
        # 添加其他 kwargs
        init_kwargs.update(kwargs)
        
        # 调用初始化
        engine = deepspeed.initialize(**init_kwargs)
        
        # 新版本返回格式可能不同
        if isinstance(engine, tuple):
            # 旧版本返回格式: (model, optimizer, _, lr_scheduler)
            return engine
        else:
            # 新版本返回 DeepSpeedEngine 对象
            # 需要提取 optimizer 和 lr_scheduler
            optimizer = getattr(engine, 'optimizer', optimizer)
            lr_scheduler = getattr(engine, 'lr_scheduler', lr_scheduler)
            return engine, optimizer, None, lr_scheduler
    else:
        # 旧版本 API (0.6.5 及以下)
        return deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            config=config,
            dist_init_required=dist_init_required if dist_init_required is not None else True,
            **kwargs
        )


def configure_activation_checkpointing(mpu, checkpoint_activations=False,
                                      deepspeed_config=None, num_checkpoints=None,
                                      **kwargs):
    """
    配置激活检查点，兼容新旧版本
    
    Args:
        mpu: 模型并行工具
        checkpoint_activations: 是否启用激活检查点
        deepspeed_config: DeepSpeed 配置
        num_checkpoints: 检查点数量
        **kwargs: 其他参数
    
    Returns:
        None
    """
    if not HAS_DEEPSPEED:
        return
    
    if mpu is None:
        return
    
    # 新版本 DeepSpeed (0.9.0+) 可能通过 config 配置激活检查点
    if is_deepspeed_version("0.9.0"):
        # 检查配置文件中是否已经配置了激活检查点
        if deepspeed_config and isinstance(deepspeed_config, dict):
            activation_config = deepspeed_config.get("activation_checkpointing", {})
            if activation_config and activation_config.get("partition_activations", False):
                # 配置文件中已配置，可能需要通过其他方式设置
                # 新版本可能不再需要手动配置
                pass
        
        # 尝试使用新的 API
        if hasattr(deepspeed, 'activation_checkpointing'):
            # 新版本的激活检查点 API
            try:
                from deepspeed.runtime.activation_checkpointing import checkpointing
                # 设置 checkpoint 函数到 mpu
                if hasattr(checkpointing, 'checkpoint'):
                    mpu.checkpoint = checkpointing.checkpoint
                if hasattr(checkpointing, 'get_cuda_rng_tracker'):
                    mpu.get_cuda_rng_tracker = checkpointing.get_cuda_rng_tracker
                if hasattr(checkpointing, 'model_parallel_cuda_manual_seed'):
                    mpu.model_parallel_cuda_manual_seed = checkpointing.model_parallel_cuda_manual_seed
                return
            except ImportError:
                pass
        
        # 如果新 API 不可用，使用 PyTorch 的梯度检查点
        if checkpoint_activations:
            try:
                from torch.utils.checkpoint import checkpoint
                mpu.checkpoint = checkpoint
                warnings.warn(
                    "Using PyTorch's gradient checkpointing instead of DeepSpeed's. "
                    "Consider configuring activation_checkpointing in DeepSpeed config file."
                )
            except ImportError:
                pass
    else:
        # 旧版本 API (0.6.5 及以下)
        if hasattr(deepspeed, 'checkpointing') and hasattr(deepspeed.checkpointing, 'configure'):
            deepspeed.checkpointing.configure(
                mpu,
                partition_activations=checkpoint_activations,
                contiguous_checkpointing=kwargs.get('contiguous_checkpointing', False),
                checkpoint_in_cpu=kwargs.get('checkpoint_in_cpu', False),
                num_checkpoints=num_checkpoints,
                synchronize=kwargs.get('synchronize', checkpoint_activations),
                profile=kwargs.get('profile', checkpoint_activations),
                deepspeed_config=deepspeed_config,
            )
            mpu.checkpoint = deepspeed.checkpointing.checkpoint
            if hasattr(deepspeed.checkpointing, 'get_cuda_rng_tracker'):
                mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            if hasattr(deepspeed.checkpointing, 'model_parallel_cuda_manual_seed'):
                mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def reset_activation_checkpointing():
    """重置激活检查点"""
    if not HAS_DEEPSPEED:
        return
    
    if hasattr(deepspeed, 'checkpointing') and hasattr(deepspeed.checkpointing, 'reset'):
        deepspeed.checkpointing.reset()


def is_activation_checkpointing_configured():
    """检查激活检查点是否已配置"""
    if not HAS_DEEPSPEED:
        return False
    
    if hasattr(deepspeed, 'checkpointing') and hasattr(deepspeed.checkpointing, 'is_configured'):
        return deepspeed.checkpointing.is_configured()
    
    return False


def get_checkpointing_functions():
    """获取检查点相关函数"""
    if not HAS_DEEPSPEED:
        return None, None, None
    
    if is_deepspeed_version("0.9.0"):
        # 新版本
        try:
            from deepspeed.runtime.activation_checkpointing import checkpointing
            checkpoint = getattr(checkpointing, 'checkpoint', None)
            get_cuda_rng_tracker = getattr(checkpointing, 'get_cuda_rng_tracker', None)
            model_parallel_cuda_manual_seed = getattr(checkpointing, 'model_parallel_cuda_manual_seed', None)
            return checkpoint, get_cuda_rng_tracker, model_parallel_cuda_manual_seed
        except ImportError:
            # 降级到 PyTorch
            try:
                from torch.utils.checkpoint import checkpoint
                return checkpoint, None, None
            except ImportError:
                return None, None, None
    else:
        # 旧版本
        if hasattr(deepspeed, 'checkpointing'):
            checkpoint = getattr(deepspeed.checkpointing, 'checkpoint', None)
            get_cuda_rng_tracker = getattr(deepspeed.checkpointing, 'get_cuda_rng_tracker', None)
            model_parallel_cuda_manual_seed = getattr(deepspeed.checkpointing, 'model_parallel_cuda_manual_seed', None)
            return checkpoint, get_cuda_rng_tracker, model_parallel_cuda_manual_seed
    
    return None, None, None

