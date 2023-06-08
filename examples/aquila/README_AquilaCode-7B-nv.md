license: [Apache License 2.0](https://model.baai.ac.cn/use-agreement)


# AquilaCode-7B-nv

## 简介/Overview
Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，在Aquila的训练过程中实现了比Magtron+DeepSpeed ZeRO-2将近８倍的训练效率。Aquila语言大模型是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。

The Aquila language model inherits the architectural design advantages of GPT-3 and LLaMA, replacing a batch of more efficient underlying operator implementations and redesigning the tokenizer for Chinese-English bilingual support. It upgrades the BMTrain parallel training method, achieving nearly 8 times the training efficiency of Magtron+DeepSpeed ZeRO-2 in the training process of Aquila. The Aquila language model is trained from scratch on high-quality Chinese and English corpora. Through data quality control and various training optimization methods, it achieves better performance than other open-source models with smaller datasets and shorter training times. It is also the first large-scale open-source language model that supports Chinese-English-Knowledge, commercial licensing, and complies with domestic data regulations.

AquilaCode-7B-nv是在Aquila-7B模型的基础上，经过代码数据的继续预训练得到的基础代码模型。此模型由智源研究院研发。在主流评测数据集上的评测结果如下

AquilaCode-7B-nv is a foundational code model obtained by further pretraining on code data based on the Aquila-7B model. It was developed by Beijing Academy of Artificial Intelligence. The evaluation results on mainstream benchmark datasets are as follows:

| 名称/Name | MMLU_Chinese_EM | CLUE-EM |MMLU-EM| BoolQ-EM| TruthfulQA-EM |IMDB-EM| RAFT-EM|
|  -----  | ----  | -----  | ----  | -----  | ----  | -----  | -----  |
| [AquilaCode-7B-nv](https://model.baai.ac.cn/model-detail/xxxxx) | 0.xxx | 0.xxx|0.xxx | 0.xxx|0.xxx |


您可以在[FlagEval基础模型评测平台](https://flageval.baai.ac.cn/#/home) 查看更多评测指标

You can view [FlagEval Model Evaluation Platform](https://flageval.baai.ac.cn/#/home) for more details



我们的模型也同时支持[Huggingface平台](hflink)

We also support [Huggingface](hflink)

## 模型细节/Model details

我们使用了一系列更高效的底层算子来辅助模型训练，其中包括参考[flash-attention](https://github.com/HazyResearch/flash-attention)的方法并替换了一些中间计算，同时还使用了RMSNorm。在此基础上，我们应用了[BMtrain](https://github.com/OpenBMB/BMTrain)技术进行轻量化的并行训练，该技术采用了数据并行、ZeRO（零冗余优化器）、优化器卸载、检查点和操作融合、通信-计算重叠等方法来优化模型训练过程。

Aquila模型所采用的tokenizer是由我们从头开始训练的，支持中英双语。与其他tokenizer的参数对比见下图：

We used a series of more efficient low-level operators to assist with model training, including methods referenced from [flash-attention](https://github.com/HazyResearch/flash-attention) and replacing some intermediate calculations, as well as using RMSNorm. Building upon this foundation, we applied the [BMtrain](https://github.com/OpenBMB/BMTrain) for lightweight parallel training, which utilizes methods such as data parallelism, ZeRO (zero redundancy optimizer), optimizer offloading, checkpoint and operation fusion, and communication-computation overlap to optimize the model training process.

The tokenizer used in the Aquila model was trained from scratch by us and supports both English and Chinese. The parameters of this tokenizer are compared to those of other tokenizers in the figure below:

| 模型/Model | 词表大小 | 说明 |英文平均tokens量| 中文平均tokens量|代码平均tokens量  |
|  -----  | ----  | -----  | ----  | -----  | ----  | 
| gpt2 | 50527 | bpe|1717.2914 | 1764.7128|2323.8167 |
| llama | 32000 | sp(bpe)|1805.6541| 1257.9891|1970.3644 |
| gpt2_new_100k | 100000 | bpe|1575.7418 | 477.4393|1679.7736 |


模型在8台8卡Nvidia A100-40G上训练14天，数据集规模为2350亿。

The model was trained on an 8 8-card Nvidia A100-40G for 14 days, and there are 235B tokens in the train set.

## 训练数据集/Training data 
AquilaCode-7B-nv训练使用了[starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)中的shell, sql，C, C++, Java, Javascript, Python, git-commits, github-issues, jupyter-scripts, jupyter-structured-text数据


The AquilaCode-7B-nv model was  supervised fine-tuning on  [starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)(shell, sql，C, C++, Java, Javascript, Python, git-commits, github-issues, jupyter-scripts, jupyter-structured-text).

 

## 使用方式/How to use

### 快速使用/Quick start

```python
import torch
import os
import argparse
import sys
from flagai import mpu
from flagai.auto_model.auto_loader import AutoLoader
import numpy as np
from flagai.model.predictor.predictor import Predictor
from pathlib import Path 
from flagai.data.tokenizer import Tokenizer
import time
import torch.distributed as dist
import json, datetime

import os

model_dir = "./checkpoints_in"
device = "cuda"

print(f"building model...")
loader = AutoLoader("lm", model_name="aquilacode-7b-nv",
                    only_download_config=True, 
                    use_cache=True, 
                    fp16=True,
                    model_dir=model_dir)

model = loader.get_model()
tokenizer = loader.get_tokenizer()

model.eval()

model.to(device)

vocab = tokenizer.get_vocab()

id2word = {v:k for k, v in vocab.items()}

predictor = Predictor(model, tokenizer)

max_new_tokens = 256

test_file = "./datasets/code_test.txt"
with open(test_file) as fin:
    prompt = '\n'+fin.read()+'\n'

input_ids = tokenizer.encode_plus_non_glm(prompt)["input_ids"][:-1]
input_length = len(input_ids)

max_length = input_length+max_new_tokens
with torch.no_grad():
    prompt = '''"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."''' + '''### Human: ''' + prompt.strip() + '''### Assistant:'''        
    res = predictor.predict_generate_randomsample(prompt, 
                                                    out_max_length=max_length, 
                                                    top_p=0.95, 
                                                    temperature=t0.7)
    print(res)
```

### 微调/Fine-tuning
#### Step 1: 配置模型
在`./checkpoints_in`里新建`aquila-7b`目录。将微调后的checkpoint，以及原始`aquila-7b`模型里的其余文件，包括`config.json`, `mergex.txt`, `vocab.json`, `special_tokens_map.json`放进去

#### Step 2: 修改参数
* 进入`/examples/aquila`目录
* 配置`hostfile`文件
* 配置`bmtrain_mgpu.sh`文件, 将`SCRIPT_FILE`改成`aquila_sft.py`
* 在`Aquila-sft.yaml`文件里更改参数 (可选)

#### Step 3: 启动微调
```
bash dist_trigger_docker.sh hostfile aquila-sft.yaml aquila-7b [实验名]
```



 ## 来源/Source

原代码可以点击此处[here](https://github.com/THUDM/GLM).

The original code can be found [here](https://github.com/THUDM/GLM).

## 证书/License

Aquila-7B开源模型使用 [智源Aquila系列模型许可协议](linkhere), 原始代码基于[Apache Licence 2.0](link)


Aquila-7B open-source model is licensed under [ BAAI Aquila Model Licence Agreement](linkhere). The source code is under [Apache Licence 2.0](link)