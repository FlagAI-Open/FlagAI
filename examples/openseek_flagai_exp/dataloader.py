"""
FlagAI 训练用 OpenSeek 数据加载与打包工具。

实现思路参考 `OpenSeek/examples/nanochat_exp/dataloader.py`，但输出改为
FlagAI `Trainer` 兼容的样本字典。
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Deque, Dict, Iterable, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset

from .dataset import parquets_iter_batched


class OpenSeekTokenDataset(IterableDataset):
    """
    将 parquet 文本流式转换为定长 token 序列。

    每个样本包含：
    - `input_ids`: 长度为 `seq_length` 的 token 序列
    - `labels`: 与 `input_ids` 对齐的下一 token 标签
    """

    def __init__(
        self,
        tokenizer,
        *,
        seq_length: int = 2048,
        split: str = "train",
        parquet_dir: Optional[str] = None,
        add_bos: bool = True,
        add_eos: bool = True,
        repeat: bool = True,
        max_samples: Optional[int] = None,
        min_tokens: int = 32,
    ):
        super().__init__()
        assert seq_length > 0, "seq_length 必须大于 0"
        assert split in {"train", "val"}, "split 仅支持 'train' 或 'val'"

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        self.parquet_dir = parquet_dir
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.repeat = repeat
        self.min_tokens = min_tokens
        self.max_samples = max_samples

        self.bos_id = getattr(tokenizer, "token_start_id", None)
        if self.bos_id is None and hasattr(tokenizer, "command_name_map"):
            bos_token = tokenizer.command_name_map.get("bos")
            self.bos_id = getattr(bos_token, "Id", None)

        self.eos_id = getattr(tokenizer, "token_end_id", None)
        if self.eos_id is None and hasattr(tokenizer, "command_name_map"):
            eos_token = tokenizer.command_name_map.get("eos")
            self.eos_id = getattr(eos_token, "Id", None)

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        token_buffer: Deque[int] = deque()
        needed = self.seq_length + 1

        def text_stream() -> Iterator[str]:
            while True:
                for batch in parquets_iter_batched(
                    split=self.split,
                    parquet_dir=self.parquet_dir,
                ):
                    for text in batch:
                        if text and text.strip():
                            yield text
                if not self.repeat:
                    break

        emitted = 0

        for text in text_stream():
            token_ids = self._encode_text(text)
            if len(token_ids) < self.min_tokens:
                continue

            token_buffer.extend(token_ids)

            while len(token_buffer) >= needed:
                window = list(itertools.islice(token_buffer, 0, needed))
                # 保留最后一个 token 作为下一段的起点
                for _ in range(self.seq_length):
                    token_buffer.popleft()

                emitted += 1
                if self.max_samples is not None and emitted > self.max_samples:
                    return

                yield {
                    "input_ids": window[:-1],
                    "labels": window[1:],
                }

        # IterableDataset: 迭代结束表示数据耗尽（非 repeat 模式）

    # ------------------------------------------------------------------ #
    # 内部工具

    def _encode_text(self, text: str) -> List[int]:
        token_ids = self.tokenizer.encode(text)
        if isinstance(token_ids, dict):
            token_ids = token_ids.get("input_ids", [])

        ids: List[int] = []
        if self.add_bos and self.bos_id is not None:
            ids.append(int(self.bos_id))

        ids.extend(int(t) for t in token_ids if t is not None)

        if self.add_eos and self.eos_id is not None:
            ids.append(int(self.eos_id))

        return ids


def build_openseek_collate_fn(
    tokenizer,
    *,
    pad_to_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> callable:
    """
    生成用于 DataLoader 的默认 collate_fn。
    """

    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "token_pad_id", 0)

    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("batch 为空，请检查 DataLoader 设置")

        seq_len = pad_to_length or len(batch[0]["input_ids"])
        inputs = []
        labels = []
        for sample in batch:
            input_ids = sample["input_ids"][:seq_len]
            label_ids = sample["labels"][:seq_len]

            if pad_to_length:
                input_ids = _pad_sequence(input_ids, seq_len, pad_token_id)
                label_ids = _pad_sequence(label_ids, seq_len, -100)
            elif len(input_ids) != seq_len or len(label_ids) != seq_len:
                raise ValueError(
                    "样本长度不一致，请开启 pad_to_length 或检查数据生成逻辑"
                )

            inputs.append(input_ids)
            labels.append(label_ids)

        input_tensor = torch.tensor(inputs, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_tensor != pad_token_id).long()

        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }

    return collate


def _pad_sequence(sequence: List[int], length: int, pad_value: int) -> List[int]:
    if len(sequence) >= length:
        return sequence[:length]
    return sequence + [pad_value] * (length - len(sequence))


