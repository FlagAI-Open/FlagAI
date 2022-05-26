# GLM 介绍

## 模型简介

目前，存在几种不同的预训练模型架构：仅实现编码器架构的自动编码模型（例如BERT），仅实现解码器的自回归模型（例如GPT），以及同时实现编码器和解码器的编码器-解码器模型（例如T5）。

[**GLM模型**](https://arxiv.org/abs/2103.10360)与这些模型略有不同。它采用了一种自回归的空白填充方法， 并且在NLP领域三种主要的任务(自然语言理解，无条件生成，有条件生成)上都取得了不错的结果。
<div align=center><img src="img/glm_example_1.png" width="600px"></div>

GLM的主要功能包括：

- 任务一：文本的一些区间会被屏蔽（参照自动编码的做法）。 这些区间将被随机重新排列，并以自动回归方式进行预测。屏蔽的区间覆盖原始文本的15%。
- 任务二：与第一个任务类似，但是区间会覆盖原始文本的50%-100%。
- 剩下GLM相对于BERT的改动
  - [Pre-LN](http://proceedings.mlr.press/v119/xiong20b.html)
  - 2D 位置编码：每个token都有两个位置编码：句子中的全局位置和屏蔽区间内的局部位置。
  - 前馈网络被线性层取代

## GLM的表现

1. 通过多任务预训练，GLM-Doc 和 GLM-Sent 的表现略逊于 GLM-Large，但仍优于 BERT-Large 和 UniLM-Large。


2. 在多任务模型中，GLM-Sent 平均优于 GLM-Doc 1.1%。将 GLM-Doc 的参数增加到 410M（1.25×BERT-Large）会得到比 GLM-Large 更好的性能。至于具有 515M 参数（1.5×BERT-Large）的 GLM 能表现得更好。

<div align=center><img src="img/glm_results2.png"></div>

1. GLM-XXLarge 的平均得分为 79.297，在多项任务中得到显着提高。在选择的3个通用+2业务评估任务中，平均提升2.47pp。

2. CLUE1.0中的任务中，除CMRC任务外，平均提升1.56pp，其中C3和OCNLI数据集提升明显（+9.9PP，+2.84PP）。

<div align=center><img src="img/glm_performance.png"></div>

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