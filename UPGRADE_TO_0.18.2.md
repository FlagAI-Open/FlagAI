# FlagAI 升级到 DeepSpeed 0.18.2 和 Megatron-LM 0.14.0 指南

## 一、概述

本文档说明如何将 FlagAI 从 DeepSpeed 0.6.5 升级到 0.18.2，以及添加 Megatron-LM 0.14.0 支持。

## 二、依赖更新

### 2.1 更新 requirements.txt

已更新 `requirements.txt`：
```txt
deepspeed>=0.18.2
megatron-core>=0.14.0
```

### 2.2 更新 setup.py

已更新 `setup.py`，添加可选依赖：
```python
extras_require={
    'deepspeed': ['deepspeed>=0.18.2'],
    'megatron': ['megatron-core>=0.14.0'],
    'all': ['deepspeed>=0.18.2', 'megatron-core>=0.14.0'],
}
```

### 2.3 安装新版本

```bash
# 安装 DeepSpeed
pip install deepspeed>=0.18.2

# 安装 Megatron-Core
pip install megatron-core>=0.14.0

# 或者使用 extras
pip install flagai[all]
```

## 三、API 兼容性处理

### 3.1 兼容性适配模块

创建了 `flagai/compat/deepspeed_compat.py` 模块，用于处理 API 兼容性：

- `initialize_deepspeed()`: 兼容新旧版本的 DeepSpeed 初始化
- `configure_activation_checkpointing()`: 兼容新旧版本的激活检查点配置
- `reset_activation_checkpointing()`: 重置激活检查点
- `is_activation_checkpointing_configured()`: 检查激活检查点是否已配置

### 3.2 训练器更新

已更新以下训练器文件以使用兼容性适配模块：

- `flagai/trainer.py`
- `flagai/trainer_v1.py` (需要更新)
- `flagai/env_trainer.py` (需要更新)
- `flagai/env_trainer_v1.py` (需要更新)

### 3.3 主要 API 变化

#### DeepSpeed 初始化

**旧版本 (0.6.5)**:
```python
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=param_groups,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    mpu=mpu,
    config=config,
    dist_init_required=True
)
```

**新版本 (0.18.2)** - 使用兼容性适配:
```python
from flagai.deepspeed_utils import initialize_deepspeed

result = initialize_deepspeed(
    model=model,
    model_parameters=param_groups,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    mpu=mpu,
    config=config,
    dist_init_required=True
)
# 处理返回值
if isinstance(result, tuple):
    model, optimizer, _, lr_scheduler = result
else:
    model = result
    optimizer = getattr(model, 'optimizer', optimizer)
    lr_scheduler = getattr(model, 'lr_scheduler', lr_scheduler)
```

#### 激活检查点配置

**旧版本 (0.6.5)**:
```python
deepspeed.checkpointing.configure(
    mpu,
    partition_activations=True,
    deepspeed_config=config,
)
mpu.checkpoint = deepspeed.checkpointing.checkpoint
mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed
```

**新版本 (0.18.2)** - 使用兼容性适配:
```python
from flagai.deepspeed_utils import configure_activation_checkpointing, get_checkpointing_functions

configure_activation_checkpointing(
    mpu=mpu,
    checkpoint_activations=True,
    deepspeed_config=config,
)

checkpoint, get_cuda_rng_tracker, model_parallel_cuda_manual_seed = get_checkpointing_functions()
if checkpoint:
    mpu.checkpoint = checkpoint
if get_cuda_rng_tracker:
    mpu.get_cuda_rng_tracker = get_cuda_rng_tracker
if model_parallel_cuda_manual_seed:
    mpu.model_parallel_cuda_manual_seed = model_parallel_cuda_manual_seed
```

## 四、配置文件迁移

### 4.1 配置文件格式变化

#### 主要变化

1. **ZeRO 配置**:
   - `cpu_offload: true` → `offload_optimizer: {device: "cpu", pin_memory: true}`
   - `offload_param` 用于 ZeRO-3 参数 offload

2. **优化器**:
   - `Adam` → `AdamW` (推荐)

3. **批次大小**:
   - 新增 `train_batch_size: "auto"`

4. **移除的配置**:
   - `zero_allow_untested_optimizer` (已废弃)

5. **新增配置**:
   - `bf16` 支持
   - `memory_breakdown` 选项

### 4.2 使用迁移脚本

提供了配置文件迁移脚本 `scripts/update_deepspeed_config.py`:

```bash
# 迁移单个配置文件
python scripts/update_deepspeed_config.py examples/Aquila/deepspeed.json

# 指定输出文件
python scripts/update_deepspeed_config.py examples/Aquila/deepspeed.json -o examples/Aquila/deepspeed_new.json

# 不备份原文件
python scripts/update_deepspeed_config.py examples/Aquila/deepspeed.json --no-backup
```

### 4.3 新配置文件示例

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
        "reduce_bucket_size": "auto",
        "reduce_scatter": true
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
    "memory_breakdown": false
}
```

## 五、Megatron-LM 集成

### 5.1 Megatron-Core

新版本使用 `megatron-core` 包，而不是完整的 `megatron-lm`：

```bash
pip install megatron-core>=0.14.0
```

### 5.2 数据工具

Megatron 的数据工具（如 `indexed_dataset`）现在可能需要通过 `megatron-core` 或独立安装。

### 5.3 MPU 模块

FlagAI 的 `mpu` 模块需要与 Megatron-Core 兼容。如果使用外部 Megatron，可能需要调整。

## 六、测试和验证

### 6.1 安装测试

```bash
# 检查 DeepSpeed 版本
python -c "import deepspeed; print(deepspeed.__version__)"

# 检查 Megatron-Core 版本
python -c "import megatron.core; print(megatron.core.__version__)"

# 运行 DeepSpeed 报告
ds_report
```

### 6.2 功能测试

1. **基本训练测试**:
   ```bash
   python examples/Aquila/aquila_pretrain.py --env_type=deepspeed
   ```

2. **模型并行测试**:
   ```bash
   python examples/Aquila/aquila_pretrain.py --env_type=deepspeed+mpu
   ```

3. **配置文件测试**:
   - 确保配置文件可以正确加载
   - 检查 ZeRO 配置是否生效
   - 检查激活检查点是否工作

### 6.3 已知问题

1. **MPU 参数**: 新版本 DeepSpeed 可能不再需要 `mpu` 参数，或通过配置方式处理
2. **检查点 API**: 激活检查点 API 可能已变化，兼容性模块会处理
3. **返回值格式**: `deepspeed.initialize()` 返回值格式可能变化

## 七、迁移步骤

### 步骤 1: 更新依赖

```bash
# 更新 requirements.txt 和 setup.py (已完成)
pip install -r requirements.txt
# 或
pip install flagai[all]
```

### 步骤 2: 迁移配置文件

```bash
# 迁移所有配置文件
find examples -name "deepspeed.json" -exec python scripts/update_deepspeed_config.py {} \;
```

### 步骤 3: 更新代码

训练器文件已更新，但可能需要测试和调整：

- `flagai/trainer.py` ✅
- `flagai/trainer_v1.py` ⚠️ (需要更新)
- `flagai/env_trainer.py` ⚠️ (需要更新)
- `flagai/env_trainer_v1.py` ⚠️ (需要更新)

### 步骤 4: 测试

```bash
# 运行测试
python -m pytest tests/

# 运行示例
python examples/Aquila/aquila_pretrain.py --env_type=deepspeed
```

## 八、回退方案

如果遇到问题，可以回退到旧版本：

```bash
# 回退 DeepSpeed
pip install deepspeed==0.6.5

# 恢复配置文件
find examples -name "deepspeed.json.backup" -exec sh -c 'mv "$1" "${1%.backup}"' _ {} \;
```

## 九、后续工作

1. **更新其他训练器**: 更新 `trainer_v1.py`, `env_trainer.py`, `env_trainer_v1.py`
2. **测试**: 全面测试所有功能
3. **文档**: 更新文档和示例
4. **性能优化**: 利用新版本的性能优化
5. **新功能**: 添加对新功能的支持（如 BF16、ZeRO-Infinity 等）

## 十、参考资源

- [DeepSpeed 0.18.2 文档](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [Megatron-Core GitHub](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed 配置参考](https://www.deepspeed.ai/docs/config-json/)
- [DeepSpeed 发布说明](https://github.com/microsoft/DeepSpeed/releases)

## 十一、支持

如有问题，请通过以下方式联系：

- GitHub Issues: https://github.com/FlagAI-Open/FlagAI/issues
- 官方邮箱: open.platform@baai.ac.cn

