## Transformer basics
简单来说，[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中首次提出的Transformers
可以看作是编码器模块和解码器模块的集成。随着注意力机制的使用，
Transformer可以从输入中提取更好的特征，这点使得基于Transformer的模型当前在大多数语言任务中有着了十分优越的表现。
Transformer的另一个特点是它与并行计算的兼容性，这是其与RNN等时序模型相比起来的优点。

Transformer的结构如下图所示
<div align=center><img src="img/transformer.png" width="400px"></div>  

在编码器步骤中，首先将输入编码与位置编码相加，
然后将相加的结果传递给多头注意力机制，该机制能够考虑不同位置token之间的相关性信息。
注意力机制计算的输出将加到原始输入里，然后添加层归一化结构。
编码器剩下的部分是有层规范化的前馈层，以及第二个相加操作。

解码器结构与编码器类似，但有以下区别：

1. 在开始时，我们需要为解码器屏蔽未来的信息，这是通过将矩阵的上三角设置为0来完成的。
2. 有一个中间层，其中包含来自编码器的查询(Q)和键(K)，以及来自解码器的值(V)。
3. 解码器在输出端有一个线性层和softmax层，以确定词汇表中每个标记的输出概率

N个编码器和解码器将堆叠在一起形成Transformer，其中N通常为12或24。

Transformer的详细结构信息可以参考 [这篇文章](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0).

## Transformer Applications

目前，存在几种不同的预训练模型体系结构：仅实现编码器体系结构的自编码模型（例如BERT），
仅实现解码器的自回归模型（例如GPT），以及同时实现编码器和解码器的编码器-解码器模型（例如T5）。

[All NLP Tasks Are Generation Tasks: A General Pretraining Framework](https://arxiv.org/abs/2103.10360) 这篇论文里提出的**GLM模型**, 声称使用新的预训练方法，从而使其在分类，
无条件生成和条件生成任务中具有良好的性能。

GLM的主要功能包括：

- 任务一：文本的一些区间会被屏蔽（参照自动编码的做法）。
这些区间将被随机重新排列，并以自动回归方式进行预测。屏蔽的区间覆盖原始文本的15%。
- 任务二：与第一个任务类似，但是区间会覆盖原始文本的50%-100%
- 剩下GLM相对于BERT的改动
  - [Pre-LN](http://proceedings.mlr.press/v119/xiong20b.html)
  - 2D 位置编码：每个token都有两个位置编码：句子中的全局位置和屏蔽区间内的局部位置。.
  - 前馈网络被线性层取代
关于GLM的自回归以及自编码操作可以查看 [这里](APPENDIX_GLM_IO.md)的示例.
