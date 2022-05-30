# 使用 FlagAI 提示学习工具包来提高在SuperGLUE任务上的表现

## 背景
提示学习(prompt-learning)是一种自然语言处理 (NLP) 领域广泛使用的范式，尤其是在数据有限的情况下表现良好。它遵循预训练-微调范式的思想：即预训练好的语言模型将被适配到各种下游任务上。不过与微调不同的是，提示学习会将原本的任务转化为一个特定的模板，这个过程被称为提示工程。

## 提示工程
提示工程中包含两种主要方法：完形填空提示和前缀提示。 完形填空提示更适合使用遮挡类语言模型的下游任务，而前缀提示通常用于文本生成任务。

### 完形填空提示
通常来说，输入文本和标签可以从任务数据集中获得。这些数据将被重新排列成一个完形填空式的短语(包括一个带有空格的段落以及空格处可以填充的选项)，这包括两个步骤：
1. 构建一个由输入部分、答案部分和一些自然语言组成的模板。
2. 构建一个从原始的标签到可填选项的对应关系。

下面是一个完形填空提示的示例

数据集包含两个输入文本：前提和假设。标签表示前提和假设之间的关系，标签有三种可能的文本字符串：entailment（一致）、contradiction（矛盾）和neutral(无偏向)。

<div align=center><img src="img/dataset_example_2.png" width="500px"></div>

我们可以设计一个如下的模板：

<div align=center><img src="img/dataset_example_3.png" width="500px"></div>

### 前缀提示
前缀提示并没有用真正的自然语言设计模板，而是直接修改嵌入空间。如下图所示，它在Transformer里插入一个前缀向量，并冻结语言模型的其余参数。与完形填空提示对比，这么做相当于从离散的tokens变成了连续的向量，避免了人工设计模板的好坏对于模型性能的影响过大的问题。
<div align=center><img src="img/prompt_figure_2.png" width="600px"></div>


## 引用

Liu, P. (2021, July 28). Pre-train, Prompt, and Predict: A Systematic Survey of Prompting. . . arXiv.Org. https://arxiv.org/abs/2107.13586 

Ding, N. (2021, November 3). OpenPrompt: An Open-source Framework for Prompt-learning. arXiv.Org. https://arxiv.org/abs/2111.01998

Schick, T. (2020, January 21). Exploiting Cloze Questions for Few Shot Text Classification and. . . arXiv.Org. https://arxiv.org/abs/2001.07676

Li, X. L. (2021, January 1). Prefix-Tuning: Optimizing Continuous Prompts for Generation. arXiv.Org. https://arxiv.org/abs/2101.00190
