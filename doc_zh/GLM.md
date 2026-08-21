# GLM 介绍

## 模型简介

目前，存在几种不同的预训练模型架构：仅实现编码器架构的自动编码模型（例如BERT），仅实现解码器的自回归模型（例如GPT），以及同时实现编码器和解码器的编码器-解码器模型（例如T5）。

[**GLM模型**](https://arxiv.org/abs/2103.10360)与这些模型略有不同。它采用了一种自回归的空白填充方法， 并且在NLP领域三种主要的任务(自然语言理解，无条件生成，有条件生成)上都取得了不错的结果。

| Framwork        | NLU | Cond.Gen. | Uncond.Gen |
|-----------------|-----|-----------|------------|
| Augoregressive  | -   | -         | ✅          |
| Autoencoding    | ✅   | ×         | ×          |
| Encoder-Decoder | -   | ✅         | -          |
| GLM             | ✅   | ✅         | ✅          |

GLM的主要功能包括：

- 任务一：文本的一些区间会被屏蔽（参照自动编码的做法）。 这些区间将被随机重新排列，并以自动回归方式进行预测。屏蔽的区间覆盖原始文本的15%。
- 任务二：与第一个任务类似，但是区间会覆盖原始文本的50%-100%。
- 剩下GLM相对于BERT的改动
  - [Pre-LN](http://proceedings.mlr.press/v119/xiong20b.html)
  - 2D 位置编码：每个token都有两个位置编码：句子中的全局位置和屏蔽区间内的局部位置。
  - 前馈网络被线性层取代

## GLM的表现


### [SuperGLUE](https://super.gluebenchmark.com)
单模型单任务微调在`dev`集上的效果，更多结果在[这里](https://github.com/THUDM/GLM)

|  Model | COPA | WSC | RTE | WiC | CB | MultiRC | BoolQ | ReCoRD |
|  ----  | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GLM-10b-ch  | 98.0 | 95.2 | 93.1 | 75.7 | 98.7/98.2 | 88.1/63.3 | 88.7 | 94.4/94.0 |
| [RoBERTa-Large](https://github.com/pytorch/fairseq/tree/master/examples/roberta) | 94.0 | 91.3 | 86.6 | 75.6 | 98.2/- | 85.7/- | 86.9 |89.5/89.0|
| [DeBERTa-XXLarge-v2](https://github.com/microsoft/DeBERTa/tree/master/experiments/superglue) | 97.0 | - | 93.5 | - | - | 87.8/63.6 | 88.3 | 94.1/93.7 |

### [CLUE](https://www.cluebenchmarks.com)
单模型单任务微调在CLUE数据集上的结果（测试还在进行中，这里列出了部分任务）。如果想要使用`GLM-10b-ch`请点[这里](https://model.baai.ac.cn/model-detail/100001)。

|      模型      |  AFQMC | TNEWS1.0 | IFLYTEK | OCNLI_50K |   CSL  | CMRC2018 | CHID1.0 | C3 1.0 |
|:--------------:|:------:|:--------:|:-------:|:---------:|:------:|:--------:|:-------:|:------:|
| RoBERTa XLarge | 75.835 |   68.75  |  62.654 |   82.333  | 83.433 |   80.5   |  86.57  |  77.03 |
|   GLM-10b-ch   |  75.42 |   69.94  |  62.15  |     85    |  86.17 |    70    |  87.009 | 88.335 |

## FlagAI支持的GLM预训练模型
参考 [Tutorial 5: 使用AutoLoader工具快速构建模型](/doc_zh/TUTORIAL_5_INSTRUCTIONS_FOR_AutoLoader.md)。

## Step-by-step procedure of GLM
1) 如下图所示，原文包含6个token，两个区间被屏蔽：第一个区间包含第3个token，第二个区间包含第5个和第6个token。

<div align=center><img src="img/glm_example_2.png" width="400px"></div>

2) 将输入分成两个部分： A 部分 (将遮挡区间遮盖掉后的文本)和B部分(被遮挡的区间). 注意所有被遮挡区间的顺序会被重新打乱。

<div align=center><img src="img/glm_example_3.png" width="400px"></div>

3) GLM 的输入和输出，输入包含输入编码和两组位置编码：第一组是每个token的位置，其中每个遮挡区间中的token共享相同的位置 ID。第二组记录token在遮挡区间内的相对位置。
<div align=center><img src="img/glm_example_4.png" width="400px"></div>

4) 下图里的自注意力机制既通过遮挡文本实现了自编码， 也在预测遮挡区间内文本的过程里实现了自回归。
<div align=center><img src="img/glm_example_5.png" width="400px"></div>