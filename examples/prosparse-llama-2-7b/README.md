---
language:
- en
library_name: transformers
license: llama2
pipeline_tag: text-generation

---


# ProSparse-LLaMA-2-7B

- Model creator: [Meta](https://huggingface.co/meta-llama)
- Original model: [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Fine-tuned by: [THUNLP](https://nlp.csai.tsinghua.edu.cn/) and [ModelBest](modelbest.cn)
- Paper: [link](https://arxiv.org/pdf/2402.13516.pdf)

### Introduction

The utilization of activation sparsity, namely the existence of considerable weakly-contributed elements among activation outputs, is a promising method for inference acceleration of large language models (LLMs) ([Liu et al., 2023](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf); [Song et al., 2023](https://arxiv.org/pdf/2312.12456.pdf)). Concretely, acceleration methods based on activation sparsity usually achieve higher inference speed by making wiser resource allocation and computation policies to avoid resource waste on these weakly-contributed parameters.

Adopting ReLU as the activation function is a straightforward method to achieve activation sparsity. However, most recent mainstream LLMs adopt activation functions without intrinsic sparsity (e.g., GELU and Swish). Some efforts ([Zhang et al., 2022](https://aclanthology.org/2022.findings-acl.71.pdf); [Mirzadeh et al., 2023](https://arxiv.org/pdf/2310.04564.pdf); [Zhang et al., 2024](https://arxiv.org/pdf/2402.03804.pdf)) introduce ReLU or its variants as the substitutive activation function to help non-ReLU LLMs achieve activation sparsity and inference acceleration, but few can concurrently obtain high sparsity and comparable task-specific performance.

In this work, we introduce a simple and effective sparsification method named "ProSparse" to push LLMs for higher activation sparsity while maintaining comparable performance. By applying ProSparse to Swish-activated LLaMA2-7B, LLaMA2-13B, and MiniCPM-1B, we obtain ReLU-activated models with high sparsity of 89.32%, 88.80%, and 87.89%, respectively, while their performance is comparable to the original version. These present the most sparsely activated models among open-source LLaMA versions and competitive end-size models, considerably surpassing ReluLLaMA-7B (66.98%) and ReluLLaMA-13B (71.56%). Further inference acceleration experiments demonstrate the practical speedup effects of higher sparsity on both [PowerInfer](https://arxiv.org/pdf/2312.12456.pdf) and our two sparse GPU [operators](https://github.com/Raincleared-Song/sparse_gpu_operator).

### Training Dataset

We train the 7B model on about 34.60 billion tokens within 16,500 steps, including a mixture of the following two categories of data.

- Language modeling datasets:

  * StarCoder

  * Wikipedia

  * Pile
  * Other collected datasets

- Instruction tuning datasets:

  - UltraChat
  - P3 (multiple-choice QA)
  - PAQ
  - Unnatural Instructions
  - Flan
  - Super-Natural Instructions
  - Other collected datasets

Intuitively, training the model with even more tokens or with data of a wider coverage and higher quality will obtain better task-specific performance.

### ProSparse: Training Methodology

The training process of ProSparse consists of three steps (refer to Section 3.2 of [paper](https://arxiv.org/pdf/2402.13516.pdf) for more details):

1. **Activation Function Substitution**: We substitute the activation function of FFNs with ReLU and apply continual training;
2. **Progressive Sparsity Regularization**: We jointly optimize the model on the conventional next-token prediction loss and \\(L_1\\) regularization loss. The regularization is applied to the sparse intermediate outputs of FFNs with a regularization factor increasing progressively in multiple stages. Specifically, the regularization factor \\(\lambda\\) is set to a small constant for the warmup stage, and then increases along a smooth sine curve for each of the subsequent incremental stages. Each stage is accompanied by certain steps of training. In this way, the model can have more time to adapt to the increasing regularization without radical activation shifts, thus alleviating performance degradation.
3. **Activation Threshold Shifting**: We finally replace ReLU with FATReLU ([Kurtz et al., 2020](https://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf)), a ReLU variant with a positive threshold. This can prune those non-zero weakly-contributed elements in activation outputs and further boost sparsity.

The 7B model is trained on 8 A100 GPUs. The learning rate (LR) is controlled by a cosine scheduler with a peak LR of \\(3e-5\\). The hyper-parameters for each stage (including the regularization factor \\(\lambda_i\\), the accumulated training steps \\(T_i\\), and the accumulated training tokens) are shown as follows:

| Step Number \\(i\\) | \\(\lambda_i\\) | \\(T_i\\)  | Accumulated Tokens (B) |
| :-------------: | :---------: | :----: | :--------------------: |
|        0        |      0      | 5,000  |         10.49          |
|        1        |   \\(5e-3\\)    | 6,000  |         12.58          |
|        2        |   \\(5e-2\\)    | 10,000 |         20.97          |
|        3        |   \\(5e-2\\)    | 12,000 |         25.17          |
|        4        |   \\(2e-1\\)    | 16,000 |         33.55          |
|        5        |   \\(2e-1\\)    | 16,500 |         34.60          |

### Evaluation Results

The evaluation results on the above benchmarks demonstrate the advantage of ProSparse, which is the only method achieving high sparsity and comparable performance to the original Swish-activated LLaMA2. Note that models under all settings are trained with the same number of tokens on the same mixed dataset. Our evaluation is based on the framework [UltraEval](https://github.com/OpenBMB/UltraEval). The evaluation details are listed as follows:

- **Code Generation**: We compute the average pass@1 scores on HumanEval (0-shot) and MBPP (3-shot).

- **Commonsense Reasoning**: We report the average 0-shot accuracies on PIQA, SIQA, HellaSwag, WinoGrande, and COPA.

- **Reading Comprehension**: We compute the average 0-shot accuracies on BoolQ, LAMBADA, and TyDi QA.

- **Other Popular Benchmarks**: We report the average accuracies on GSM8K (8-shot), MMLU (5-shot), Big Bench Hard (BBH) (3-shot), and AGI-Eval (0-shot).

**Notes**: For PIQA, SIQA, HellaSwag, WinoGrande, COPA, BoolQ, LAMBADA, TyDi QA, and AGI-Eval, we obtain the predicted answers based on maximized perplexity. For GSM8K, MMLU, and BBH, the predicted answers are directly generated.

|        Setting        | Average<br>Sparsity | Average<br>Performance | Code<br>Generation | Commonsense<br>Reasoning | Reading<br>Comprehension | GSM8K | MMLU  |  BBH  | AGI Eval |
| :-------------------: | :----------------: | :----------------------: | :----------------------: | :---: | :---: | :---: | :---------: | :-----: | :-----------------: |
| LLaMA2-7B    | - | 37.96 | 16.37 | 69.59 | 61.87 | 12.96 | 44.45 | 32.96 | 27.53 |
| ReluLLaMA-7B | 66.98 | 37.62 | 15.85 | 69.64 | 70.54 |  5.84 | 38.64 | 35.07 | 27.73 |
| **ProSparse-7B**\* | 88.11 | 38.31 | 19.47 | 66.29 | 63.33 | 12.74 | 45.21 | 33.59 | 27.55 |
| **ProSparse-7B**   | **89.32** | **38.46** | 19.42 | 66.27 | 63.50 | 12.13 | 45.48 | 34.99 | 27.46 |
| LLaMA2-13B | - | 44.06 | 20.19 | 72.58 | 71.55 | 22.21 | 54.69 | 37.89 | 29.33 |
| ReluLLaMA-13B | 71.56 | 42.74 | 20.19 | 70.44 | 73.29 | 18.50 | 50.58 | 37.97 | 28.22 |
| **ProSparse-13B**\* | 87.97 | **45.07** | 29.03 | 69.75 | 67.54 | 25.40 | 54.78 | 40.20 | 28.76 |
| **ProSparse-13B**   | **88.80** | 44.90 | 28.42 | 69.76 | 66.91 | 26.31 | 54.35 | 39.90 | 28.67 |
| MiniCPM-1B | - | 44.44 | 36.85 | 63.67 | 60.90 | 35.48 | 50.44 | 35.03 | 28.71 |
| **ProSparse-1B**\*  | 86.25 | **44.72** | 41.38 | 64.55 | 60.69 | 34.72 | 49.36 | 34.04 | 28.27 |
| **ProSparse-1B**    | **87.89** | **44.72** | 42.04 | 64.37 | 60.73 | 34.57 | 49.51 | 34.08 | 27.77 |

**Notes**: "Original" refers to the original Swish-activated LLaMA2 versions. ReluLLaMA-7B and ReluLLaMA-13B are available at [7B](https://huggingface.co/SparseLLM/ReluLLaMA-7B) and [13B](https://huggingface.co/SparseLLM/ReluLLaMA-13B) respectively. MiniCPM-1B is available at [1B](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16). "ProSparse-7B\*", "ProSparse-13B\*", and "ProSparse-1B\*" denote the ProSparse versions without activation threshold shifting.

### Evaluation Issues with LM-Eval

The above results can be replicated with [UltraEval](https://github.com/OpenBMB/UltraEval). Some abnormal results obtained with other popular frameworks such as [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness) are probably attributed to the absence of the cls token `<s>`, which is not added by default in LM-Eval. A quick temporary fix is shown in the following codes. Other differences in evaluation results may be caused by other reasons, including the few-shot settings, data pre-processing, and extra prompts.

```python
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L945
for _, context_enc, continuation_enc in chunk:
    # sanity check
    assert len(context_enc) > 0
    # Note: a trivial fix here
    if context_enc[0] != 1:
        context_enc = [1] + context_enc
    assert len(continuation_enc) > 0
    assert len(continuation_enc) <= self.max_length
```

Here are the steps to adapting the original [vLLM](https://github.com/vllm-project/vllm) to ProSparse LLaMA models.

1. Replace the file [vllm/model_executor/models/llama.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py) in original vLLM with this [file](https://github.com/Raincleared-Song/DejaVu_predictor/blob/main/llama.py).
2. Replace the contents of the original [config.json](https://huggingface.co/SparseLLM/prosparse-llama-2-7b/blob/main/config.json) with this [file](https://github.com/Raincleared-Song/DejaVu_predictor/blob/main/config.json).
3. Set the environment variable `ACT_INFO`. To test the version without activation threshold shifting, `export ACT_INFO=relu`. To test the version with activation threshold shifting, `export ACT_INFO=fatrelu_0.01`.

### Inference Acceleration Effects

First, we utilize [PowerInfer](https://arxiv.org/pdf/2312.12456.pdf), a state-of-the-art acceleration framework leveraging activation sparsity. As its inference speed and accuracy heavily rely on the performance of activation predictors, we report the activation recall and predicted sparsity (i.e., two key metrics for evaluating the activation predictor) as well as the number of tokens generated per second by PowerInfer (with one A100 GPU and sufficient CPUs). The GGUF files and activation predictors for ProSparse-7B are available at [ProSparse-LLaMA-2-7B-GGUF](https://huggingface.co/PowerInfer/prosparse-llama-2-7b-gguf) ([duplicate](https://huggingface.co/SparseLLM/prosparse-llama-2-7b-gguf)) and [ProSparse-LLaMA-2-7B-Predictor](https://huggingface.co/PowerInfer/prosparse-llama-2-7b-predictor) ([duplicate](https://huggingface.co/SparseLLM/prosparse-llama-2-7b-predictor)) respectively.

Moreover, considering the potential inference inaccuracies caused by wrong predictions of activation predictors, we implement two sparse GPU [operators](https://github.com/Raincleared-Song/sparse_gpu_operator) for faster accurate inference utilizing activation sparsity. They are responsible for the speedup of two key steps in a gated FFN:

- Step (2) (`S2`): a fused operator of ReLU and \\(\mathbf{s} \odot (\mathbf{x} \mathbf{W}_1^T)\\);
- Step (3) (`S3`): a sparse matrix-vector multiplication operator \\(\mathbf{x}_1 \mathbf{W}_2^T\\).

where \\(\mathbf{s}\\), \\(\mathbf{x}\\), \\(\mathbf{x}_1\\), and \\(\odot\\) denote the gating scores, the FFN input hidden states, the intermediate outputs, and the element-wise multiplication respectively. \\(\mathbf{W}_1\\) and \\(\mathbf{W}_2\\) are FFN weight matrices.

The acceleration effects of LLMs with different sparsity are displayed as follows. ProSparse, which reaches a high sparsity without performance degradation, can gain the most benefits among all the settings concerned. Refer to Section 4.3 of [paper](https://arxiv.org/pdf/2402.13516.pdf) for more details.

|        Setting        | Average<br>Sparsity | Activation<br>Recall | Predicted<br>Sparsity | PowerInfer<br>Speed | Speedup<br>to Dense | `S2`<br>Time | Speedup<br>to Dense | `S3`<br/>Time | Speedup<br/>to Dense |
| :-------------------: | :-----------------: | :------------------: | :-------------------: | :-----------------: | :-----------------: | :--------------: | :-----------------: | :---------------: | :------------------: |
| Dense-7B | - | - | - | 3.67 | 1.00 | 90.55 | 1.00 | 82.92 | 1.00 |
|     ReluLLaMA-7B      |        66.98        |        90.89         |         58.95         |        11.37        | 3.10 |      67.12       |        1.35         |       63.00       |         1.32         |
| **ProSparse-7B**\*  |        88.11        |      **93.46**       |         75.24         |        **16.30**        | **4.44** |      46.66       |        1.94         |       55.56       |         1.49         |
|   **ProSparse-7B**    |      **89.32**      |        92.34         |       **78.75**       |          -          | - |      **45.38**       |        **2.00**         |       **55.05**       |         **1.51**         |
| Dense-13B | - | - | - | 1.92 | 1.00 | 131.36 | 1.00 | 113.68 | 1.00 |
|     ReluLLaMA-13B     |        71.56        |        86.41         |         71.93         |        6.59         | 3.43 |      69.92       |        1.88         |       75.47       |         1.51         |
| **ProSparse-13B**\* |        87.97        |        91.02         |         77.93         |        **8.67**         | **4.52** |      55.29       |        2.38         |       67.50       |         1.68         |
|   **ProSparse-13B**   |        **88.80**        |        **91.11**         |         **78.28**         |          -          | - |      **53.78**       |        **2.44**         |       **66.73**       |         **1.70**         |

**Notes**: For "Dense" settings, the "Inference Speed" (token/sec) is obtained by [llama.cpp](https://github.com/ggerganov/llama.cpp), and the time (us) for steps (2) and (3) is measured without sparse GPU operators. For other sparse settings, the "Inference Speed" is obtained by [PowerInfer](https://arxiv.org/pdf/2312.12456.pdf), and sparse GPU operators are applied. ProSparse settings with activation threshold shifting and the MiniCPM architecture are not supported by PowerInfer at present.

### License Disclaimer

This model is bound by the license & usage restrictions of the original Llama-2 model and comes with no warranty or guarantees of any kind.

### Limitations & Biases

Llama 2 and fine-tuned variants are a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Llama 2 and any fine-tuned variant's potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts. Therefore, before deploying any applications of Llama 2 variants, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available at https://ai.meta.com/llama/responsible-use-guide/

### Citation

Please kindly cite using the following BibTeX:

```bibtex
@article{song2024prosparse,
  title={{ProSparse}: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models},
  author={Song, Chenyang and Han, Xu and Zhang, Zhengyan and Hu, Shengding and Shi, Xiyu and Li, Kuai and Chen, Chen and Liu, Zhiyuan and Li, Guangli and Yang, Tao and Sun, Maosong},
  year={2024},
  journal={arXiv preprint arXiv:2402.13516},
  url={https://arxiv.org/pdf/2402.13516.pdf}
}
```

#### Acknowledgments

The model card is modified from [ReluLLaMA-7B](https://huggingface.co/SparseLLM/ReluLLaMA-7B).
