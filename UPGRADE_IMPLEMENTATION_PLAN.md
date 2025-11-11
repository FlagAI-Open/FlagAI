# FlagAI 升级实施计划 - 详细代码修复指南

## 一、DeepSpeed API 兼容性修复

### 1.1 DeepSpeed 初始化 API 修复

#### 当前代码问题
**文件**: `flagai/trainer.py` (第 506-516 行)

```python
# 当前代码 (可能不兼容新版本)
if 'deepspeed' in self.env_type:
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        mpu=mpu if self.env_type == 'deepspeed+mpu' else None,
        config=self.deepspeed_config,
        dist_init_required=True)
```

#### 修复方案

**方案 A: 版本兼容处理**
```python
import deepspeed
import packaging.version

def initialize_deepspeed(model, model_parameters, optimizer, lr_scheduler, 
                         mpu, config, env_type):
    """兼容新旧版本的 DeepSpeed 初始化"""
    ds_version = packaging.version.parse(deepspeed.__version__)
    
    # 新版本 (0.9.0+) 可能不再需要 dist_init_required
    if ds_version >= packaging.version.parse("0.9.0"):
        init_kwargs = {
            "model": model,
            "model_parameters": model_parameters,
            "config": config,
        }
        
        if optimizer is not None:
            init_kwargs["optimizer"] = optimizer
        if lr_scheduler is not None:
            init_kwargs["lr_scheduler"] = lr_scheduler
        if env_type == 'deepspeed+mpu' and mpu is not None:
            init_kwargs["mpu"] = mpu
            
        # 新版本可能返回不同的值
        engine = deepspeed.initialize(**init_kwargs)
        
        # 新版本返回 DeepSpeedEngine 对象
        if hasattr(engine, 'module'):
            return engine, engine.optimizer, None, engine.lr_scheduler
        else:
            # 兼容旧版本返回格式
            return engine
    else:
        # 旧版本 API
        return deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mpu=mpu if env_type == 'deepspeed+mpu' else None,
            config=config,
            dist_init_required=True)

# 使用
if 'deepspeed' in self.env_type:
    result = initialize_deepspeed(
        model=model,
        model_parameters=param_groups,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        mpu=mpu if self.env_type == 'deepspeed+mpu' else None,
        config=self.deepspeed_config,
        env_type=self.env_type
    )
    if isinstance(result, tuple):
        model, optimizer, _, lr_scheduler = result
    else:
        model = result
        optimizer = model.optimizer
        lr_scheduler = model.lr_scheduler
```

**方案 B: 使用最新 API (推荐)**
```python
# 新版本 DeepSpeed (0.9.0+) 推荐方式
if 'deepspeed' in self.env_type:
    # 准备初始化参数
    ds_config = self.deepspeed_config
    
    # 创建 DeepSpeed 引擎
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_config,
    )
    
    # 如果有 MPU，需要特殊处理
    if self.env_type == 'deepspeed+mpu' and mpu is not None:
        # 新版本可能通过 config 配置 MPU
        # 或者需要单独的设置
        pass
    
    model = engine
    self.model = model
    self.optimizer = optimizer
```

### 1.2 Checkpointing API 修复

#### 当前代码问题
**文件**: `flagai/trainer.py` (第 312-319 行)

```python
# 当前代码
if 'deepspeed' in self.env_type and self.deepspeed_activation_checkpointing:
    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=self.checkpoint_activations,
        contiguous_checkpointing=False,
        checkpoint_in_cpu=False,
        num_checkpoints=None,
        synchronize=self.checkpoint_activations,
        profile=self.checkpoint_activations,
        deepspeed_config=self.deepspeed_config,
    )
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed
```

#### 修复方案

```python
def configure_activation_checkpointing(mpu, checkpoint_activations, 
                                      deepspeed_config, env_type):
    """配置激活检查点，兼容新旧版本"""
    import deepspeed
    import packaging.version
    
    ds_version = packaging.version.parse(deepspeed.__version__)
    
    if env_type != 'deepspeed+mpu':
        return
    
    # 新版本 DeepSpeed (0.9.0+) 可能通过 config 配置
    if ds_version >= packaging.version.parse("0.9.0"):
        # 方式 1: 通过配置文件配置
        if deepspeed_config and 'activation_checkpointing' in deepspeed_config:
            # 已经在配置文件中配置，无需额外设置
            pass
        else:
            # 方式 2: 使用新的 API
            if hasattr(deepspeed, 'activation_checkpointing'):
                # 新版本的激活检查点配置
                if checkpoint_activations:
                    # 可能需要使用梯度检查点函数
                    from deepspeed.runtime.activation_checkpointing import checkpointing
                    mpu.checkpoint = checkpointing.checkpoint
            else:
                # 降级到 PyTorch 的梯度检查点
                from torch.utils.checkpoint import checkpoint
                mpu.checkpoint = checkpoint
    else:
        # 旧版本 API
        deepspeed.checkpointing.configure(
            mpu,
            partition_activations=checkpoint_activations,
            contiguous_checkpointing=False,
            checkpoint_in_cpu=False,
            num_checkpoints=None,
            synchronize=checkpoint_activations,
            profile=checkpoint_activations,
            deepspeed_config=deepspeed_config,
        )
        mpu.checkpoint = deepspeed.checkpointing.checkpoint
        if hasattr(deepspeed.checkpointing, 'get_cuda_rng_tracker'):
            mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
        if hasattr(deepspeed.checkpointing, 'model_parallel_cuda_manual_seed'):
            mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed

# 使用
if 'deepspeed' in self.env_type and self.deepspeed_activation_checkpointing:
    configure_activation_checkpointing(
        mpu=mpu if self.env_type == 'deepspeed+mpu' else None,
        checkpoint_activations=self.checkpoint_activations,
        deepspeed_config=self.deepspeed_config,
        env_type=self.env_type
    )
```

### 1.3 训练步骤 API 修复

#### 当前代码问题
**文件**: `flagai/trainer.py` (第 795-843 行)

```python
# 当前代码
def train_step_deepspeed(self, data, model, optimizer, lr_scheduler, mems=None, single_step=False):
    # ...
    if (self.accumulate_count + 1) % self.gradient_accumulation_steps == 0:
        model.set_gradient_accumulation_boundary(True)
    else:
        model.set_gradient_accumulation_boundary(False)
    # ...
    model.backward(lm_loss)
    # ...
    if lr_scheduler:
        lr_scheduler.step()
    # ...
    if (self.accumulate_count + 1) % self.gradient_accumulation_steps == 0:
        self.accumulate_count = 0
        model.step()
    else:
        self.accumulate_count += 1
```

#### 修复方案

```python
def train_step_deepspeed(self, data, model, optimizer, lr_scheduler, mems=None, single_step=False):
    """兼容新旧版本的训练步骤"""
    import deepspeed
    import packaging.version
    
    ds_version = packaging.version.parse(deepspeed.__version__)
    
    # 前向传播
    self.timers('forward').start()
    step_output = self.forward_step(data, model, mems)
    self.timers('forward').stop()
    lm_loss = step_output['loss']
    
    # 损失归约
    reduced_loss = lm_loss.detach().clone().view(1)
    if self.env_type == 'deepspeed+mpu':
        torch.distributed.all_reduce(reduced_loss.data,
                                     group=mpu.get_data_parallel_group())
    elif self.env_type == 'deepspeed':
        torch.distributed.all_reduce(reduced_loss.data)
    
    if 'deepspeed' in self.env_type:
        reduced_loss.data = reduced_loss.data / \
            (self.world_size / self.model_parallel_size)
    
    if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
        # 梯度累积设置
        if ds_version >= packaging.version.parse("0.9.0"):
            # 新版本可能在配置中设置梯度累积
            # 或者使用不同的 API
            if hasattr(model, 'set_gradient_accumulation_boundary'):
                if (self.accumulate_count + 1) % self.gradient_accumulation_steps == 0:
                    model.set_gradient_accumulation_boundary(True)
                else:
                    model.set_gradient_accumulation_boundary(False)
        else:
            # 旧版本
            if (self.accumulate_count + 1) % self.gradient_accumulation_steps == 0:
                model.set_gradient_accumulation_boundary(True)
            else:
                model.set_gradient_accumulation_boundary(False)
        
        # 反向传播
        self.timers('backward').start()
        model.backward(lm_loss)
        self.timers('backward').stop()
        
        # 优化器步骤
        self.timers('optimizer').start()
        
        # 学习率调度器
        if lr_scheduler and (self.accumulate_count + 1) % self.gradient_accumulation_steps == 0:
            if ds_version >= packaging.version.parse("0.9.0"):
                # 新版本可能自动处理
                if hasattr(model, 'lr_scheduler') and model.lr_scheduler:
                    model.lr_scheduler.step()
                elif lr_scheduler:
                    lr_scheduler.step()
            else:
                if lr_scheduler:
                    lr_scheduler.step()
        
        # 更新参数
        if (self.accumulate_count + 1) % self.gradient_accumulation_steps == 0:
            self.accumulate_count = 0
            model.step()
        else:
            self.accumulate_count += 1
        
        self.timers('optimizer').stop()
        dist.barrier()
    else:
        log_dist("Found NaN loss, skip backward", [0])
        del lm_loss, reduced_loss
        mems = []
        reduced_loss = None
    
    return reduced_loss, mems
```

## 二、DeepSpeed 配置文件更新

### 2.1 配置文件迁移

#### 旧配置文件示例
**文件**: `examples/Aquila/deepspeed.json`

```json
{
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 100,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": false,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e7,
        "allgather_bucket_size": 5e7,
        "cpu_offload": true
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 2000
        }
    },
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-5,
            "weight_decay": 0.1,
            "betas": [0.9, 0.98],
            "eps": 1e-6
        }
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": false
    },
    "wall_clock_breakdown": false
}
```

#### 新配置文件示例

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "weight_decay": 0.1,
            "betas": [0.9, 0.98],
            "eps": 1e-6
        }
    },
    
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 2000
        }
    },
    
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    "bf16": {
        "enabled": false
    },
    
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    
    "wall_clock_breakdown": false,
    "memory_breakdown": false,
    
    "flops_profiler": {
        "enabled": false,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 3,
        "detailed": true
    }
}
```

#### 配置迁移脚本

```python
#!/usr/bin/env python3
"""
DeepSpeed 配置文件迁移脚本
将旧版本配置文件迁移到新版本格式
"""

import json
import sys
import os
from pathlib import Path

def migrate_deepspeed_config(old_config_path, new_config_path=None):
    """迁移 DeepSpeed 配置文件"""
    with open(old_config_path, 'r') as f:
        old_config = json.load(f)
    
    new_config = {}
    
    # 基本配置
    if "train_micro_batch_size_per_gpu" in old_config:
        new_config["train_micro_batch_size_per_gpu"] = old_config["train_micro_batch_size_per_gpu"]
        new_config["train_batch_size"] = "auto"
    
    if "gradient_accumulation_steps" in old_config:
        new_config["gradient_accumulation_steps"] = old_config["gradient_accumulation_steps"]
    
    if "gradient_clipping" in old_config:
        new_config["gradient_clipping"] = old_config["gradient_clipping"]
    
    if "steps_per_print" in old_config:
        new_config["steps_per_print"] = old_config["steps_per_print"]
    
    # ZeRO 优化配置
    if "zero_optimization" in old_config:
        zero_config = old_config["zero_optimization"].copy()
        new_zero_config = {
            "stage": zero_config.get("stage", 0),
            "overlap_comm": zero_config.get("overlap_comm", True),
            "contiguous_gradients": zero_config.get("contiguous_gradients", True),
        }
        
        # CPU offload 配置迁移
        if zero_config.get("cpu_offload", False):
            new_zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True
            }
            if zero_config.get("stage", 0) >= 2:
                new_zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
        
        # 其他 ZeRO 配置
        if "reduce_bucket_size" in zero_config:
            new_zero_config["reduce_bucket_size"] = "auto"  # 或保留原值
        if "allgather_bucket_size" in zero_config:
            # 新版本可能不再需要
            pass
        
        new_config["zero_optimization"] = new_zero_config
    
    # 优化器配置
    if "optimizer" in old_config:
        new_config["optimizer"] = old_config["optimizer"].copy()
        # 确保使用 AdamW 而不是 Adam
        if new_config["optimizer"].get("type") == "Adam":
            new_config["optimizer"]["type"] = "AdamW"
    
    # 学习率调度器
    if "scheduler" in old_config:
        new_config["scheduler"] = old_config["scheduler"]
    
    # FP16 配置
    if "fp16" in old_config:
        fp16_config = old_config["fp16"].copy()
        new_fp16_config = {
            "enabled": fp16_config.get("enabled", False),
            "auto_cast": False,
            "loss_scale": fp16_config.get("loss_scale", 0),
            "initial_scale_power": 16,
            "loss_scale_window": fp16_config.get("loss_scale_window", 1000),
            "hysteresis": fp16_config.get("hysteresis", 2),
            "min_loss_scale": fp16_config.get("min_loss_scale", 1),
        }
        new_config["fp16"] = new_fp16_config
    
    # BF16 配置（新增）
    new_config["bf16"] = {
        "enabled": False
    }
    
    # 激活检查点配置
    if "activation_checkpointing" in old_config:
        act_config = old_config["activation_checkpointing"].copy()
        new_act_config = {
            "partition_activations": act_config.get("partition_activations", False),
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": act_config.get("contiguous_memory_optimization", False),
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        }
        new_config["activation_checkpointing"] = new_act_config
    
    # 其他配置
    if "wall_clock_breakdown" in old_config:
        new_config["wall_clock_breakdown"] = old_config["wall_clock_breakdown"]
    
    new_config["memory_breakdown"] = False
    
    # 移除废弃的配置
    # "zero_allow_untested_optimizer" 在新版本中已移除
    
    # 保存新配置
    if new_config_path is None:
        new_config_path = old_config_path.replace('.json', '_new.json')
    
    with open(new_config_path, 'w') as f:
        json.dump(new_config, f, indent=4)
    
    print(f"迁移完成: {old_config_path} -> {new_config_path}")
    return new_config_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python migrate_config.py <old_config.json> [new_config.json]")
        sys.exit(1)
    
    old_config_path = sys.argv[1]
    new_config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    migrate_deepspeed_config(old_config_path, new_config_path)
```

## 三、Megatron 集成改进

### 3.1 处理 Megatron 数据工具依赖

#### 问题代码
**文件**: `flagai/data/dataset/indexed_dataset/build_datasets.py`

```python
# 当前代码 - 直接导入外部 Megatron
from megatron import mpu, print_rank_0
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.gpt_dataset import _build_shuffle_idx, _build_doc_idx, _num_epochs, _num_tokens, get_indexed_dataset_, _build_sample_idx
```

#### 修复方案

**方案 A: 使用 FlagAI 内部的实现**
```python
# 修改为使用 FlagAI 内部实现
try:
    # 优先使用 FlagAI 内部实现
    from flagai.mpu import get_model_parallel_rank, get_model_parallel_world_size
    from flagai.data.dataset.indexed_dataset.dataset_utils import (
        get_train_valid_test_split_,
        get_datasets_weights_and_num_samples,
    )
    from flagai.data.dataset.indexed_dataset.indexed_dataset import (
        make_dataset as make_indexed_dataset,
    )
    from flagai.data.dataset.indexed_dataset.gpt_dataset import (
        _build_shuffle_idx,
        _build_doc_idx,
        _num_epochs,
        _num_tokens,
        get_indexed_dataset_,
        _build_sample_idx,
    )
    from flagai.logger import log_dist as print_rank_0
except ImportError:
    # 降级到外部 Megatron（如果可用）
    try:
        from megatron import mpu, print_rank_0
        from megatron.data.dataset_utils import get_train_valid_test_split_
        from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
        from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
        from megatron.data.gpt_dataset import _build_shuffle_idx, _build_doc_idx, _num_epochs, _num_tokens, get_indexed_dataset_, _build_sample_idx
    except ImportError:
        raise ImportError(
            "需要 Megatron 或 FlagAI 内部实现。"
            "请安装 Megatron-LM 或确保 FlagAI 包含必要的实现。"
        )
```

**方案 B: 添加可选依赖**
```python
# 在 setup.py 中添加可选依赖
extras_require = {
    "megatron": ["megatron-lm>=2.0.0"],  # 或具体的 Megatron 版本
    "deepspeed": ["deepspeed>=0.12.0"],
    "all": ["deepspeed>=0.12.0", "megatron-lm>=2.0.0"],
}

# 在代码中检查
try:
    import megatron
    HAS_MEGATRON = True
except ImportError:
    HAS_MEGATRON = False

if HAS_MEGATRON:
    from megatron import mpu, print_rank_0
    # ...
else:
    # 使用 FlagAI 内部实现
    from flagai.mpu import ...
    # ...
```

### 3.2 改进 MPU 模块

#### 当前实现问题
**文件**: `flagai/mpu/initialize.py`

当前实现只支持基本的 Tensor 并行，缺少 Pipeline 并行等高级功能。

#### 改进方案

```python
# flagai/mpu/initialize.py 改进版本

import torch
from .utils import ensure_divisibility

# 全局变量
_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP = None

def initialize_model_parallel(
    tensor_model_parallel_size_=1,
    pipeline_model_parallel_size_=1,
    virtual_pipeline_model_parallel_size_=None,
):
    """
    初始化模型并行组
    
    Args:
        tensor_model_parallel_size_: Tensor 并行大小
        pipeline_model_parallel_size_: Pipeline 并行大小
        virtual_pipeline_model_parallel_size_: 虚拟 Pipeline 并行大小
    """
    if torch.distributed.get_rank() == 0:
        print(f'> 初始化模型并行: '
              f'Tensor={tensor_model_parallel_size_}, '
              f'Pipeline={pipeline_model_parallel_size_}')
    
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    
    # 验证并行大小
    ensure_divisibility(world_size, tensor_model_parallel_size_ * pipeline_model_parallel_size_)
    
    rank = torch.distributed.get_rank()
    
    # Tensor 模型并行组
    global _TENSOR_MODEL_PARALLEL_GROUP
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size_
    num_pipeline_model_parallel_groups = world_size // (tensor_model_parallel_size_ * pipeline_model_parallel_size_)
    
    # 构建 Tensor 并行组
    for i in range(num_pipeline_model_parallel_groups):
        start_rank = i * pipeline_model_parallel_size_ * tensor_model_parallel_size_
        end_rank = (i + 1) * pipeline_model_parallel_size_ * tensor_model_parallel_size_
        for j in range(tensor_model_parallel_size_):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size_)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group
    
    # Pipeline 模型并行组
    global _PIPELINE_MODEL_PARALLEL_GROUP
    if pipeline_model_parallel_size_ > 1:
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size_ * pipeline_model_parallel_size_,
                         (i + 1) * tensor_model_parallel_size_ * pipeline_model_parallel_size_)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
    
    # 数据并行组
    global _DATA_PARALLEL_GROUP
    num_data_parallel_groups = world_size // (tensor_model_parallel_size_ * pipeline_model_parallel_size_)
    for i in range(tensor_model_parallel_size_ * pipeline_model_parallel_size_):
        ranks = [i + j * tensor_model_parallel_size_ * pipeline_model_parallel_size_
                for j in range(num_data_parallel_groups)]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
    
    # 模型并行组（Tensor + Pipeline）
    global _MODEL_PARALLEL_GROUP
    for i in range(num_data_parallel_groups):
        ranks = [i * tensor_model_parallel_size_ * pipeline_model_parallel_size_ + j
                for j in range(tensor_model_parallel_size_ * pipeline_model_parallel_size_)]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

def get_tensor_model_parallel_group():
    """获取 Tensor 模型并行组"""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_pipeline_model_parallel_group():
    """获取 Pipeline 模型并行组"""
    return _PIPELINE_MODEL_PARALLEL_GROUP

def get_data_parallel_group():
    """获取数据并行组"""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP

def get_model_parallel_group():
    """获取模型并行组（Tensor + Pipeline）"""
    return _MODEL_PARALLEL_GROUP

# 其他辅助函数...
```

## 四、测试和验证

### 4.1 创建测试脚本

```python
#!/usr/bin/env python3
"""
DeepSpeed 和 Megatron 升级测试脚本
"""

import torch
import deepspeed
import sys
import os

def test_deepspeed_import():
    """测试 DeepSpeed 导入"""
    print("测试 DeepSpeed 导入...")
    try:
        import deepspeed
        print(f"✓ DeepSpeed 版本: {deepspeed.__version__}")
        return True
    except ImportError as e:
        print(f"✗ DeepSpeed 导入失败: {e}")
        return False

def test_deepspeed_initialize():
    """测试 DeepSpeed 初始化 API"""
    print("测试 DeepSpeed 初始化 API...")
    try:
        import deepspeed
        import inspect
        
        # 检查 initialize 函数签名
        sig = inspect.signature(deepspeed.initialize)
        print(f"✓ deepspeed.initialize 签名: {sig}")
        
        # 检查参数
        params = list(sig.parameters.keys())
        print(f"  参数: {params}")
        
        return True
    except Exception as e:
        print(f"✗ DeepSpeed 初始化 API 测试失败: {e}")
        return False

def test_checkpointing_api():
    """测试 Checkpointing API"""
    print("测试 Checkpointing API...")
    try:
        import deepspeed
        
        # 检查 checkpointing 模块
        if hasattr(deepspeed, 'checkpointing'):
            print("✓ deepspeed.checkpointing 存在")
            print(f"  属性: {dir(deepspeed.checkpointing)}")
        else:
            print("✗ deepspeed.checkpointing 不存在")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Checkpointing API 测试失败: {e}")
        return False

def test_megatron_import():
    """测试 Megatron 导入"""
    print("测试 Megatron 导入...")
    try:
        # 尝试导入 Megatron
        try:
            import megatron
            print(f"✓ Megatron 可用")
            return True
        except ImportError:
            print("⚠ Megatron 未安装（可选）")
            return True  # Megatron 是可选的
    except Exception as e:
        print(f"✗ Megatron 导入测试失败: {e}")
        return False

def test_mpu_module():
    """测试 MPU 模块"""
    print("测试 MPU 模块...")
    try:
        from flagai import mpu
        print("✓ FlagAI MPU 模块可用")
        print(f"  函数: {[x for x in dir(mpu) if not x.startswith('_')]}")
        return True
    except Exception as e:
        print(f"✗ MPU 模块测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("FlagAI DeepSpeed/Megatron 升级测试")
    print("=" * 60)
    print()
    
    results = []
    
    # 运行测试
    results.append(("DeepSpeed 导入", test_deepspeed_import()))
    results.append(("DeepSpeed 初始化 API", test_deepspeed_initialize()))
    results.append(("Checkpointing API", test_checkpointing_api()))
    results.append(("Megatron 导入", test_megatron_import()))
    results.append(("MPU 模块", test_mpu_module()))
    
    # 打印结果
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    # 总结
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败，请检查上述错误")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## 五、实施步骤总结

### 步骤 1: 准备环境
1. 创建升级分支
2. 安装最新 DeepSpeed
3. 运行测试脚本

### 步骤 2: API 兼容性修复
1. 修复 DeepSpeed 初始化 API
2. 修复 Checkpointing API
3. 修复训练步骤 API
4. 测试基本功能

### 步骤 3: 配置文件迁移
1. 运行配置迁移脚本
2. 更新所有示例配置文件
3. 测试配置加载

### 步骤 4: Megatron 集成
1. 处理数据工具依赖
2. 改进 MPU 模块
3. 更新模型并行层
4. 测试并行训练

### 步骤 5: 功能测试
1. 运行单元测试
2. 运行集成测试
3. 运行示例代码
4. 性能测试

### 步骤 6: 文档更新
1. 更新 README
2. 更新配置文档
3. 更新迁移指南
4. 更新示例代码

## 六、注意事项

1. **向后兼容**: 尽量保持现有 API 不变
2. **版本检查**: 添加版本兼容性检查
3. **错误处理**: 添加详细的错误信息
4. **文档**: 及时更新文档
5. **测试**: 充分测试所有功能
6. **性能**: 确保性能不下降
7. **稳定性**: 确保稳定性

## 七、参考资源

- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed 配置参考](https://www.deepspeed.ai/docs/config-json/)
- [DeepSpeed API 文档](https://deepspeed.readthedocs.io/)

