"""
FlagAI + OpenSeek dataset conversion utilities.

本模块参考 `OpenSeek/examples/nanochat_exp/dataset.py`，用于将 OpenSeek 数据
转换为 FlagAI 训练可用的 parquet 分片格式。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset, load_dataset
except ImportError as exc:
    raise ImportError(
        "请先安装依赖：pip install pyarrow datasets huggingface_hub"
    ) from exc


DEFAULT_DATASET_NAME = "BAAI/OpenSeek-Pretrain-100B"


# -----------------------------------------------------------------------------
# 路径与目录

def resolve_data_dir(explicit_dir: Optional[str] = None) -> Path:
    """获取存放 parquet 分片的基础目录。"""
    if explicit_dir is not None:
        base_dir = Path(explicit_dir).expanduser()
    else:
        env_dir = os.environ.get("FLAGAI_OPENSEEK_DATA_DIR")
        if env_dir:
            base_dir = Path(env_dir).expanduser()
        else:
            base_dir = Path.home() / ".cache" / "flagai_openseek"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def resolve_parquet_dir(base_dir: Optional[str] = None) -> Path:
    """返回 parquet 分片目录并自动创建。"""
    root = resolve_data_dir(base_dir)
    parquet_dir = root / "parquet_shards"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    return parquet_dir


# -----------------------------------------------------------------------------
# 数据加载与转换

def load_openseek_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    *,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    trust_remote_code: bool = True,
):
    """
    从 HuggingFace 或本地目录加载 OpenSeek 数据集。
    """
    print(f"[dataset] 准备加载数据集：{dataset_name}")
    try:
        dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        )
        return dataset
    except Exception as exc:
        print(f"[dataset] 远程加载失败：{exc}")
        print("[dataset] 尝试从本地目录读取...")
        local_path = Path(dataset_name)
        if local_path.exists():
            dataset = load_dataset(
                str(local_path),
                cache_dir=cache_dir,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
            )
            return dataset
        raise


def convert_to_parquet(
    dataset,
    *,
    output_dir: Optional[str] = None,
    shard_size: int = 250_000_000,
    text_column: str = "text",
    max_shards: int = -1,
) -> int:
    """
    将 OpenSeek 数据集转换为 parquet 文件。
    """
    parquet_dir = Path(output_dir) if output_dir else resolve_parquet_dir()
    parquet_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    current_texts: List[str] = []
    current_chars = 0

    def write_shard(texts: List[str], index: int):
        filename = parquet_dir / f"shard_{index:05d}.parquet"
        table = pa.Table.from_arrays([pa.array(texts)], names=["text"])
        pq.write_table(table, filename, compression="snappy")
        print(
            f"[dataset] 写出分片 {index:05d} -> {filename.name} "
            f"({len(texts)} 样本，约 {current_chars:,} 字符)"
        )

    print(f"[dataset] 输出目录：{parquet_dir}")
    print(f"[dataset] 目标分片字符数：~{shard_size:,}")

    iterable: Iterable = dataset
    if isinstance(dataset, dict):
        iterable = dataset.get("train", list(dataset.values())[0])
        print(f"[dataset] 选用 split：{getattr(iterable, 'split', 'train')}")

    for example in iterable:
        text = example.get(text_column, "") if isinstance(example, dict) else ""
        if not text or not text.strip():
            continue

        current_texts.append(text)
        current_chars += len(text)

        if current_chars >= shard_size:
            write_shard(current_texts, shard_index)
            shard_index += 1
            current_texts = []
            current_chars = 0

            if max_shards > 0 and shard_index >= max_shards:
                break

    if current_texts:
        write_shard(current_texts, shard_index)
        shard_index += 1

    print(f"[dataset] 转换完成！共生成 {shard_index} 个分片。")
    return shard_index


# -----------------------------------------------------------------------------
# parquet 工具函数

def list_parquet_files(parquet_dir: Optional[str] = None) -> List[Path]:
    """返回指定目录下排序后的 parquet 文件列表。"""
    target_dir = Path(parquet_dir) if parquet_dir else resolve_parquet_dir()
    if not target_dir.exists():
        return []
    files = [
        file_path
        for file_path in sorted(target_dir.iterdir())
        if file_path.suffix == ".parquet" and not file_path.name.endswith(".tmp")
    ]
    return files


def parquets_iter_batched(
    *,
    split: str = "train",
    start: int = 0,
    step: int = 1,
    parquet_dir: Optional[str] = None,
) -> Iterator[List[str]]:
    """
    以批次方式迭代 parquet 文件，兼容分布式训练。
    """
    assert split in {"train", "val"}, "split 仅支持 'train' 或 'val'"
    parquet_files = list_parquet_files(parquet_dir)
    if not parquet_files:
        raise RuntimeError(
            "未找到任何 parquet 文件，请先执行数据转换：\n"
            "  python -m examples.openseek_flagai_exp.dataset --convert"
        )

    files_to_iterate = (
        parquet_files[:-1] if split == "train" else parquet_files[-1:]
    )

    for file_path in files_to_iterate:
        parquet_file = pq.ParquetFile(file_path)
        for row_group_index in range(start, parquet_file.num_row_groups, step):
            row_group = parquet_file.read_row_group(row_group_index, columns=["text"])
            yield row_group.column("text").to_pylist()


# -----------------------------------------------------------------------------
# CLI

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 OpenSeek 数据集转换为 FlagAI 可用的 parquet 数据。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="HuggingFace 数据集名称或本地路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="parquet 输出目录，默认为 FLAGAI_OPENSEEK_DATA_DIR/parquet_shards",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=-1,
        help="最多生成的分片数，-1 表示不限制",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=250_000_000,
        help="单个分片目标字符数，默认约 2.5e8",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="文本列名称，默认 text",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="是否启用 datasets 的 streaming 模式（适合大规模数据）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="datasets 下载缓存目录",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("FlagAI + OpenSeek 数据转换工具")
    print("=" * 60)
    print(f"数据集：{args.dataset}")
    print(f"输出目录：{args.output_dir or resolve_parquet_dir()}")
    print(f"分片字符数：~{args.shard_size:,}")
    print(f"最大分片数：{args.num_shards if args.num_shards > 0 else '无限制'}")
    print(f"Streaming：{args.streaming}")
    print("=" * 60)

    dataset = load_openseek_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )

    start_time = time.time()
    num_shards = convert_to_parquet(
        dataset,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        text_column=args.text_column,
        max_shards=args.num_shards,
    )
    duration = time.time() - start_time

    print(f"\n转换完成，共生成 {num_shards} 个分片，耗时 {duration:.2f} 秒。")


if __name__ == "__main__":
    main()


