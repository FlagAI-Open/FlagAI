"""
FlagAI + OpenSeek 预训练示例脚本。

该脚本参考 `OpenSeek/examples/nanochat_exp/train_wrapper.py`，重写以适配
FlagAI 的 `Trainer` 训练流程。
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer

from .dataloader import OpenSeekTokenDataset, build_openseek_collate_fn
from .dataset import (
    convert_to_parquet,
    load_openseek_dataset,
    resolve_data_dir,
    resolve_parquet_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 FlagAI + OpenSeek 数据进行语言模型预训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型与训练参数
    parser.add_argument("--model-name", type=str, default="opt-125m", help="AutoLoader 模型名称")
    parser.add_argument("--model-dir", type=str, default=None, help="模型权重缓存目录")
    parser.add_argument("--env-type", type=str, default="pytorch", help="Trainer 环境类型")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiment", type=str, default="openseek_flagai_pretrain", help="实验名/日志前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="单卡 batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--seq-length", type=int, default=2048, help="训练序列长度")
    parser.add_argument("--lr", type=float, default=2e-4, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("--warmup", type=float, default=0.01, help="学习率 warmup 占比")
    parser.add_argument("--fp16", action="store_true", help="是否启用半精度")
    parser.add_argument("--log-interval", type=int, default=10, help="日志输出间隔（step）")
    parser.add_argument("--eval-interval", type=int, default=200, help="验证间隔（step）")
    parser.add_argument("--save-interval", type=int, default=1000, help="保存间隔（step）")
    parser.add_argument("--max-train-samples", type=int, default=None, help="限制单 epoch 样本数量（None 表示不限制）")
    parser.add_argument("--max-val-samples", type=int, default=4096, help="验证最多使用的样本数量，0 表示关闭验证")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数量")

    # 数据路径与转换
    parser.add_argument("--data-dir", type=str, default=None, help="覆盖 FLAGAI_OPENSEEK_DATA_DIR")
    parser.add_argument("--parquet-dir", type=str, default=None, help="直接指定 parquet 目录")
    parser.add_argument("--convert-dataset", action="store_true", help="若 parquet 不存在则执行转换")
    parser.add_argument("--dataset-name", type=str, default="BAAI/OpenSeek-Pretrain-100B", help="HuggingFace 数据集名或本地路径")
    parser.add_argument("--num-shards", type=int, default=32, help="转换时生成的最大分片数")
    parser.add_argument("--shard-size", type=int, default=250_000_000, help="目标分片字符数")
    parser.add_argument("--text-column", type=str, default="text", help="数据集中使用的文本列名")
    parser.add_argument("--streaming", action="store_true", help="以 streaming 模式加载 HuggingFace 数据")
    parser.add_argument("--cache-dir", type=str, default=None, help="datasets 缓存目录")

    # Tokenizer 选项
    parser.add_argument("--no-bos", action="store_true", help="禁用 tokenizer BOS")
    parser.add_argument("--no-eos", action="store_true", help="禁用 tokenizer EOS")
    parser.add_argument("--repeat-train", action="store_true", help="是否在单 epoch 内循环遍历训练集")
    parser.add_argument("--min-tokens", type=int, default=32, help="过滤过短文本的最小 token 数")

    return parser.parse_args()


def maybe_convert_dataset(args: argparse.Namespace, parquet_dir: str) -> None:
    if not args.convert_dataset:
        return
    from .dataset import list_parquet_files
    existing_files = list_parquet_files(parquet_dir)
    if existing_files:
        print(f"[train] 检测到现有 parquet 分片（{len(existing_files)} 个），跳过转换。")
        return

    print("[train] 执行数据转换流程 ...")
    dataset = load_openseek_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )
    convert_to_parquet(
        dataset,
        output_dir=parquet_dir,
        shard_size=args.shard_size,
        text_column=args.text_column,
        max_shards=args.num_shards,
    )


def build_trainer(args: argparse.Namespace) -> Trainer:
    trainer = Trainer(
        env_type=args.env_type,
        pytorch_device=args.device,
        experiment_name=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warm_up=args.warmup,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        fp16=args.fp16,
    )
    return trainer


def build_dataloaders(
    args: argparse.Namespace,
    tokenizer,
    parquet_dir: Optional[str],
) -> tuple[DataLoader, Optional[DataLoader]]:
    train_dataset = OpenSeekTokenDataset(
        tokenizer,
        seq_length=args.seq_length,
        split="train",
        parquet_dir=parquet_dir,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
        repeat=args.repeat_train,
        min_tokens=args.min_tokens,
        max_samples=args.max_train_samples,
    )
    collate_fn = build_openseek_collate_fn(
        tokenizer,
        pad_to_length=args.seq_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=("cuda" in str(args.device).lower()),
    )

    val_loader: Optional[DataLoader] = None
    if args.max_val_samples and args.max_val_samples > 0:
        val_dataset = OpenSeekTokenDataset(
            tokenizer,
            seq_length=args.seq_length,
            split="val",
            parquet_dir=parquet_dir,
            add_bos=not args.no_bos,
            add_eos=not args.no_eos,
            repeat=False,
            min_tokens=args.min_tokens,
            max_samples=args.max_val_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=("cuda" in str(args.device).lower()),
        )

    return train_loader, val_loader


def main():
    args = parse_args()

    if args.data_dir:
        os.environ["FLAGAI_OPENSEEK_DATA_DIR"] = args.data_dir

    data_root = resolve_data_dir(args.data_dir)
    parquet_dir = (
        args.parquet_dir or str(resolve_parquet_dir(str(data_root)))
    )
    os.makedirs(parquet_dir, exist_ok=True)

    maybe_convert_dataset(args, parquet_dir)

    # 准备模型
    print("[train] 初始化模型与 tokenizer ...")
    auto_loader = AutoLoader(
        task_name="lm",
        model_name=args.model_name,
        model_dir=args.model_dir,
    )
    model = auto_loader.get_model()
    tokenizer = auto_loader.get_tokenizer()

    # 构建 DataLoader
    train_loader, val_loader = build_dataloaders(args, tokenizer, parquet_dir)

    # 初始化 Trainer 并启动训练
    trainer = build_trainer(args)

    print("[train] 开始训练 ...")
    trainer.train(
        model=model,
        train_dataset=train_loader,
        valid_dataset=val_loader,
        collate_fn=None,
        metric_methods=[],
    )


if __name__ == "__main__":
    main()


