# ALM 1.0

[English](README.md)

## 简介

ALM 模型是基于自回归填空的通用阿拉伯语预训练模型，相关参数如下：

| Name    | Params | Layers | Hidden Size | FFN Hidden size | Heads | Head Size |
| ------- | ------ | ------ | ----------- | --------------- | ----- | --------- |
| ALM 1.0 | 335M   | 24     | 1024        | 4096            | 16    | 64        |

## 训练数据集

ALM-1.0使用了全球最大的开源阿语数据集ArabicText 2022，详细信息可参看：[ArabicText 2022](https://data.baai.ac.cn/details/ArabicText-2022)

## 使用方式

### 微调

依托于[FlagAI](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FBAAI-Open%2FFlagAI)，ALM 可以用于常见的Seq2seq 任务。

### 快速使用/Quick start

- `examples/ALM/train.py`提供了使用ALM做摘要、内容生成等Seq2seq任务的微调样例。

- `examples/ALM/generate.py`提供了使用GLM模型做句子预测的样例。
