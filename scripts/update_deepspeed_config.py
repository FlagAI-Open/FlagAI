#!/usr/bin/env python3
"""
DeepSpeed 配置文件迁移脚本
将旧版本 (0.6.5) 配置文件迁移到新版本 (0.18.2) 格式
"""

import json
import sys
import os
from pathlib import Path
import argparse


def migrate_deepspeed_config(old_config_path, new_config_path=None, backup=True):
    """
    迁移 DeepSpeed 配置文件从 0.6.5 格式到 0.18.2 格式
    
    Args:
        old_config_path: 旧配置文件路径
        new_config_path: 新配置文件路径（如果为 None，则覆盖原文件）
        backup: 是否备份原文件
    """
    with open(old_config_path, 'r') as f:
        old_config = json.load(f)
    
    new_config = {}
    
    # 基本配置
    if "train_micro_batch_size_per_gpu" in old_config:
        new_config["train_micro_batch_size_per_gpu"] = old_config["train_micro_batch_size_per_gpu"]
        new_config["train_batch_size"] = "auto"
    elif "train_batch_size" in old_config:
        new_config["train_batch_size"] = old_config["train_batch_size"]
    else:
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
        
        # CPU offload 配置迁移 (旧版本 cpu_offload -> 新版本 offload_optimizer/offload_param)
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
        
        # ZeRO-3 特定配置
        if zero_config.get("stage", 0) >= 3:
            new_zero_config["stage3_prefetch_bucket_size"] = "auto"
            new_zero_config["stage3_param_persistence_threshold"] = "auto"
            new_zero_config["stage3_max_live_parameters"] = 1e9
            new_zero_config["stage3_max_reuse_distance"] = 1e9
            new_zero_config["stage3_gather_16bit_weights_on_model_save"] = True
        
        # reduce_bucket_size 和 allgather_bucket_size
        if "reduce_bucket_size" in zero_config:
            # 新版本支持 "auto"
            if isinstance(zero_config["reduce_bucket_size"], (int, float)):
                if zero_config["reduce_bucket_size"] == 5e7:
                    new_zero_config["reduce_bucket_size"] = "auto"
                else:
                    new_zero_config["reduce_bucket_size"] = zero_config["reduce_bucket_size"]
            else:
                new_zero_config["reduce_bucket_size"] = "auto"
        
        # allgather_bucket_size 在新版本中可能不再需要，或者合并到其他配置
        
        # reduce_scatter
        if "reduce_scatter" in zero_config:
            new_zero_config["reduce_scatter"] = zero_config["reduce_scatter"]
        
        new_config["zero_optimization"] = new_zero_config
    
    # 优化器配置
    if "optimizer" in old_config:
        opt_config = old_config["optimizer"].copy()
        # 确保使用 AdamW 而不是 Adam (新版本推荐)
        if opt_config.get("type") == "Adam":
            opt_config["type"] = "AdamW"
        new_config["optimizer"] = opt_config
    
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
    
    # BF16 配置（新增，默认关闭）
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
    else:
        new_config["wall_clock_breakdown"] = False
    
    new_config["memory_breakdown"] = False
    
    # 移除废弃的配置
    # "zero_allow_untested_optimizer" 在新版本中已移除，不再需要
    
    # 保存新配置
    if new_config_path is None:
        new_config_path = old_config_path
    
    # 备份原文件
    if backup and new_config_path == old_config_path:
        backup_path = old_config_path + ".backup"
        with open(backup_path, 'w') as f:
            json.dump(old_config, f, indent=4)
        print(f"原配置文件已备份到: {backup_path}")
    
    # 保存新配置
    with open(new_config_path, 'w') as f:
        json.dump(new_config, f, indent=4)
    
    print(f"配置文件迁移完成: {old_config_path} -> {new_config_path}")
    return new_config_path


def main():
    parser = argparse.ArgumentParser(description='迁移 DeepSpeed 配置文件')
    parser.add_argument('config_file', type=str, help='配置文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件路径（默认覆盖原文件）')
    parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"错误: 配置文件不存在: {args.config_file}")
        sys.exit(1)
    
    migrate_deepspeed_config(
        args.config_file,
        args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()

