# Seq2seq方法

## 编码器模型
我们提供了encoder模型来执行seq2seq任务，例如，Bert、Roberta、GLM等等。
在训练过程中，我们在编码器模型中添加了一个特殊的 attention mask。(https://github.com/microsoft/unilm)
![encoder_mask](../docs/img/encoder_mask.png)
该模型的输入是两个句子：[cls]句子1[sep]句子2[sep]。
其中，句子_1不使用mask，而句子_2使用自回归mask。

## 解码器模型
我们还为seq2seq任务提供了decoder模型，例如gpt-2模型。
![decoder_mask](../docs/img/decoder_mask.png)
给出一个起始文本，这个模型可以很好地延续文本。

## 编解码器模型
我们还为seq2seq任务提供encoder-decoder模型，例如T5模型。
![encoder_decoder_mask](../docs/img/encoder_decoder_mask.png)
encoder只需编码一次即可获得特征编码，decoder根据自身和特征编码继续生成。