# ALM 1.0

[简体中文](README_zh.md)

## Overview

The Arabic Language Model (ALM) 1.0 is a pretrained language model based on autoregressive blank infilling . Below shows the count of model parameters in detail.

| Name    | Params | Layers | Hidden Size | FFN Hidden size | Heads | Head Size |
| ------- | ------ | ------ | ----------- | --------------- | ----- | --------- |
| ALM 1.0 | 335M   | 24     | 1024        | 4096            | 16    | 64        |

## Training data

ALM-1.0 uses the largest open-source Arabic text dataset ArabicText 2022. You can check [ArabicText 2022](https://data.baai.ac.cn/details/ArabicText-2022) for more information.

## How to use

### Finetune

With [FlagAI](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FBAAI-Open%2FFlagAI), one can use ALM model directly for Seq2Seq finetuning.

### Quick start

- `examples/ALM/train.py` provides examples to use ALM for Seq2seq finetuning task, such as text summarization and short/long text generation. 

- `examples/ALM/generate.py` provides examples to use ALM for masked text prediction in an autoregressive way.
