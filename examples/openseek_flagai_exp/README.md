# FlagAI × OpenSeek 训练示例

本目录参考 `OpenSeek/examples/nanochat_exp` 的实现方式，将 **OpenSeek 预训练数据** 接入 **FlagAI Trainer**，提供从数据转换、加载到训练的完整脚本。

## 目录结构

```
examples/openseek_flagai_exp/
├── __init__.py                 # Python 包入口
├── dataset.py                  # OpenSeek → parquet 转换工具
├── dataloader.py               # FlagAI 训练用 IterableDataset & collate
├── train_flagai_openseek.py    # FlagAI Trainer 训练脚本
└── README.md                   # 本说明
```

## 前置条件

1. **准备环境**
   - Python ≥ 3.10，推荐使用虚拟环境。
   - 安装 FlagAI 及所需依赖：
     ```bash
     pip install -r requirements.txt  # FlagAI 仓库根目录自带
     pip install pyarrow datasets huggingface_hub
     ```

2. **下载模型权重**
   - 脚本默认加载 `opt-125m`，FlagAI 会自动下载对应权重。若处于离线环境，请提前将模型放到 `--model-dir` 指定目录。

3. **准备 OpenSeek 数据**
   - 推荐直接使用 HuggingFace 自动下载（默认数据集：`BAAI/OpenSeek-Pretrain-100B`）。
   - 若使用本地离线数据，请确保目录结构与官方 parquet 输出一致。

## 快速上手

### 1. 转换数据为 parquet

```bash
# 方式一：直接转换至默认缓存目录 ~/.cache/flagai_openseek
python -m examples.openseek_flagai_exp.dataset \
  --dataset BAAI/OpenSeek-Pretrain-100B \
  --num-shards 32 \
  --streaming

# 方式二：指定输出目录
export FLAGAI_OPENSEEK_DATA_DIR=/data/openseek_flagai
python -m examples.openseek_flagai_exp.dataset \
  --dataset /path/to/OpenSeek-Pretrain-100B \
  --output-dir /data/openseek_flagai/parquet_shards \
  --num-shards 64
```

转换结果默认放置在：

```
$FLAGAI_OPENSEEK_DATA_DIR/
└── parquet_shards/
    ├── shard_00000.parquet
    ├── shard_00001.parquet
    └── ...
```

### 2. 启动训练

```bash
python -m examples.openseek_flagai_exp.train_flagai_openseek \
  --model-name opt-125m \
  --batch-size 2 \
  --seq-length 2048 \
  --epochs 1 \
  --convert-dataset \
  --num-shards 16 \
  --max-train-samples 500
```

> `--convert-dataset` 会在检测到 parquet 目录为空时自动调用上一步的转换流程。  
> `--max-train-samples` / `--max-val-samples` 可用于快速验证流水线是否正常。

### 常用参数

| 参数 | 说明 |
| ---- | ---- |
| `--data-dir` | 覆盖 `FLAGAI_OPENSEEK_DATA_DIR`，便于切换数据磁盘。 |
| `--parquet-dir` | 直接指定 parquet 目录，跳过自动拼接 `parquet_shards`。 |
| `--repeat-train` | 在单 epoch 内循环遍历训练数据（默认关闭，避免无限循环）。 |
| `--no-bos` / `--no-eos` | 控制是否添加 tokenizer 的起止符。 |
| `--fp16` | 启用半精度训练（需 GPU 支持）。 |

## 设计说明

- `dataset.py` 与 OpenSeek 仓库保持一致的 CLI，输出 parquet 分片，便于跨项目共享。
- `dataloader.py` 实现 `IterableDataset`，按顺序拼接文本并切分为固定长度 token 序列，输出 FlagAI `Trainer` 期望的 `dict`。
- `train_flagai_openseek.py` 通过 `AutoLoader` 获取模型 & tokenizer，对接 `Trainer` 的统一训练循环。脚本中保留了 warmup、梯度累积等关键参数，可按需扩展。

## 验证与调试

```bash
# 检查数据目录下的分片数量
python - <<'PY'
from examples.openseek_flagai_exp.dataset import list_parquet_files
files = list_parquet_files()
print(f"Parquet files: {len(files)}")
PY

# 快速跑通一个 batch
python -m examples.openseek_flagai_exp.train_flagai_openseek \
  --batch-size 1 \
  --max-train-samples 4 \
  --max-val-samples 0 \
  --epochs 1
```

## 注意事项

- OpenSeek 数据量巨大，建议先使用较小的 `--num-shards` + `--max-train-samples` 做 sanity check，再扩大规模。
- 如果需要分布式训练，可将 `env-type` 设置为 `pytorchDDP` / `deepspeed`，并根据 FlagAI 官方文档配置额外参数。
- 训练过程中模型及缓存均可能占用大量磁盘/显存，请提前规划资源。

欢迎在运行过程中将遇到的问题反馈到仓库 Issue。🎯


