# Introduction to GLM

## Model Description

Currently, there exist several different pre-training model architectures: autoencoding models that only implement encoder architecture (e.g., BERT),
autoregressive models that only implement decoder (e.g., GPT), and encoder-decoder models that implement both encoder and decoder (e.g., T5).

The **GLM model**, proposed in [GLM: General Language Model Pretraining
with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360), is based on a slightly different strategy: autoregressive blank infilling. 

It claims to perform well in the three main categories of NLP taks: classification, unconditional generation, and conditional generation tasks.
<div align=center><img src="img/glm_example_1.png" width="600px"></div>
The key features of GLM include:

- First task: Several spans of the text are masked following the idea of autoencoding. Those spans will be randomly rearranged and be predicted in an autoregressive manner. The masked spans covers 15% original tokens.
- Second task: Similar to the first task, but the span covers 50%-100% original tokens.
- Other model architecture changes compared with BERT
  - [Pre-LN](http://proceedings.mlr.press/v119/xiong20b.html)
  - 2D positional encoding: Each token has two positional encodings: the global position in the sentence, and the local position inside the masked span.
  - Feed-forward network is replaced with a linear layer

## Performance of GLM
### SuperGLUE

1. With multi-task pretraining, GLM-Doc and GLM-Sent perform slightly worse than GLM-Large, but still outperform BERT-Large and UniLM-Large. 
2. Among multitask models, GLM-Sent outperforms GLM-Doc by 1.1% on average. Increasing GLM-Doc’s parameters to 410M (1.25×BERT-Large) leads to better performance than GLM-Large. GLM with 515M parameters (1.5×BERT-Large) can perform even better.

<div align=center><img src="img/glm_results2.png"></div>

### CLUE

1. GLM-XXLarge has has achieved an avarage score of 79.297 which is improved significantly in multiple tasks. In the selected 3 general +2 business evaluation tasks, the average improvement is 2.47pp.
2. Among the tasks in the CLUE1.0, except for the CMRC task, the average improvement is 1.56pp, of which the C3 and OCNLI data sets have improved significantly (+9.9PP, +2.84PP).


<div align=center><img src="img/glm_performance.png"></div>

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
