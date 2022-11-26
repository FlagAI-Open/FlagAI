# Introduction to GLM

## Model Description

Currently, there exist several different pre-training model architectures: autoencoding models that only implement encoder architecture (e.g., BERT),
autoregressive models that only implement decoder (e.g., GPT), and encoder-decoder models that implement both encoder and decoder (e.g., T5).

The **GLM model**, proposed in [GLM: General Language Model Pretraining
with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360), is based on a slightly different strategy: autoregressive blank infilling. 

It claims to perform well in the three main categories of NLP taks: classification, unconditional generation, and conditional generation tasks.

| Framwork        | NLU | Cond.Gen. | Uncond.Gen |
|-----------------|-----|-----------|------------|
| Augoregressive  | -   | -         | ✅          |
| Autoencoding    | ✅   | ×         | ×          |
| Encoder-Decoder | -   | ✅         | -          |
| GLM             | ✅   | ✅         | ✅          |

The key features of GLM include:

- First task: Several spans of the text are masked following the idea of autoencoding. Those spans will be randomly rearranged and be predicted in an autoregressive manner. The masked spans covers 15% original tokens.
- Second task: Similar to the first task, but the span covers 50%-100% original tokens.
- Other model architecture changes compared with BERT
  - [Pre-LN](http://proceedings.mlr.press/v119/xiong20b.html)
  - 2D positional encoding: Each token has two positional encodings: the global position in the sentence, and the local position inside the masked span.
  - Feed-forward network is replaced with a linear layer

## Performance of GLM

### [SuperGLUE](https://super.gluebenchmark.com)
Single model, single task finetune on `dev`，more  results [here](https://github.com/THUDM/GLM)

|  Model | COPA | WSC | RTE | WiC | CB | MultiRC | BoolQ | ReCoRD |
|  ----  | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GLM-10b-ch  | 98.0 | 95.2 | 93.1 | 75.7 | 98.7/98.2 | 88.1/63.3 | 88.7 | 94.4/94.0 |
| [RoBERTa-Large](https://github.com/pytorch/fairseq/tree/master/examples/roberta) | 94.0 | 91.3 | 86.6 | 75.6 | 98.2/- | 85.7/- | 86.9 |89.5/89.0|
| [DeBERTa-XXLarge-v2](https://github.com/microsoft/DeBERTa/tree/master/experiments/superglue) | 97.0 | - | 93.5 | - | - | 87.8/63.6 | 88.3 | 94.1/93.7 |

### [CLUE](https://www.cluebenchmarks.com)
Single model, single task finetune on `dev`, (list part of results on CLU). Learn more about `GLM-10b-ch` click [here](https://model.baai.ac.cn/model-detail/100001)。

|      模型      |  AFQMC | TNEWS1.0 | IFLYTEK | OCNLI_50K |   CSL  | CMRC2018 | CHID1.0 | C3 1.0 |
|:--------------:|:------:|:--------:|:-------:|:---------:|:------:|:--------:|:-------:|:------:|
| RoBERTa XLarge | 75.835 |   68.75  |  62.654 |   82.333  | 83.433 |   80.5   |  86.57  |  77.03 |
|   GLM-10b-ch   |  75.42 |   69.94  |  62.15  |     85    |  86.17 |    70    |  87.009 | 88.335 |


## Supported pre-trained GLM models
see [Tutorial 5: Simplify model and tokenizer Initialization by Using Autoloader](/docs/TUTORIAL_5_INSTRUCTIONS_FOR_AutoLoader.md).

## Step-by-step procedure of GLM
1) Following the example in the paper, the original text contains 6 tokens, and two spans are masked: first span contains the 3rd token and the second span contains the 5th and 6th token.

<div align=center><img src="img/glm_example_2.png" width="400px"></div>

2) The input is divided into 2 parts, part A (corrupted text) and part B (masked spans). Note that the order of spans is shuffled here.
3) 
<div align=center><img src="img/glm_example_3.png" width="400px"></div>

3) Input and output of GLM, the input contains token embeddings and 2 sets of positional encodings: the first set is the positions of each token, where the tokens in each masked span share the same position ID. The second set records the relative positions inside the masked span.
<div align=center><img src="img/glm_example_4.png" width="400px"></div>

4) The self-attention mask that realizes both autoencoding upon corrupted text and autoregressive upon the masked spans.
<div align=center><img src="img/glm_example_5.png" width="400px"></div>
