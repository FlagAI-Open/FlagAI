![FlagAI](logo.png)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6052/badge)](https://bestpractices.coreinfrastructure.org/projects/6052)
[![Python application](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml/badge.svg)](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/FlagAI-Open/FlagAI?include_prereleases&style=social)
[简体中文](README_zh.md)

--------------------------------------------------------------------------------


FlagAI (Fast LArge-scale General AI models) is a fast, easy-to-use and extensible toolkit for large-scale model. Our goal is to support training, fine-tuning, and deployment of large-scale models on various downstream tasks with multi-modality. 

* Now it supports text-image representation model [**AltCLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) and text-to-image generation [**AltDiffusion**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/BAAI/bilingual_stable_diffusion). And it supports **WuDao GLM** with a maximum of 10 billion parameters (see [Introduction to GLM](/docs/GLM.md)). It also supports [**EVA-CLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP), **OPT**, **BERT**, **RoBERTa**, **GPT2**, **T5**, **ALM**, and models from Huggingface Transformers.

* It provides APIs to quickly download and use those pre-trained models on a given text, fine-tune them on widely-used datasets collected from [SuperGLUE](https://super.gluebenchmark.com/) and [CLUE](https://github.com/CLUEbenchmark/CLUE) benchmarks, and then share them with the community on our model hub. It also provides [prompt-learning](/docs/TUTORIAL_7_PROMPT_LEARNING.md) toolkit for few-shot tasks.   

* These models can be applied to (Chinese/English) Text, for tasks like text classification, information extraction, question answering, summarization, and text generation.

* FlagAI is backed by the three most popular data/model parallel libraries — [PyTorch](https://pytorch.org/)/[Deepspeed](https://www.deepspeed.ai/)/[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — with seamless integration between them. Users can parallel their training/testing process with less than ten lines of code.

The code is partially based on [GLM](https://github.com/THUDM/GLM), [Transformers](https://github.com/huggingface/transformers) and [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM).

## News
- [28 Nov 2022] release v1.5.0, support 1.1B [**EVA-CLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP) and [ALM: A large Arabic Language Model based on GLM], examples in [**ALM**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/ALM)
- [10 Nov 2022] release v1.4.0, support [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v1), examples in [**AltCLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) and [**AltDiffusion**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion)
- [29 Aug 2022] release v1.3.0, Added CLIP module and redesigned tokenizer APIs in [#81](https://github.com/FlagAI-Open/FlagAI/pull/81)
- [21 Jul 2022] release v1.2.0, ViTs are supported in [#71](https://github.com/FlagAI-Open/FlagAI/pull/71)
- [29 Jun 2022] release v1.1.0, support OPTs downloading and inference/fine-tuning [#63](https://github.com/FlagAI-Open/FlagAI/pull/63)
- [17 May 2022] made our first contribution in [#1](https://github.com/FlagAI-Open/FlagAI/pull/1)

--------------------------------------------------------------------------------

<!-- toc -->

- [Requirements and Installation](#requirements-and-installation)
- [Quick Started](#quick-start)
    - [Load model and tokenizer](#load-model-and-tokenizer)
    - [Predictor](#predictor)
    - [Text-to-image generation task](/examples/AltDiffusion/README.md)
- [Pre-trained Models and examples](#pretrained-models-and-examples)
- [Tutorials](#tutorials)
- [Contributing](#contributing)
- [Contact us](#contact-us)
- [License](#license)

<!-- tocstop -->
## Requirements and Installation
* PyTorch version >= 1.8.0
* Python version >= 3.8
* For training/testing models on GPUs, you'll also need to install CUDA and NCCL

To install FlagAI with pip:
```shell
pip install -U flagai
```

- [Optional]To install FlagAI and develop locally:

```shell
git clone https://github.com/FlagAI-Open/FlagAI.git
python setup.py install
```

- [Optional] For faster training, install NVIDIA's [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- [Optional] For ZeRO optimizers, install [DEEPSPEED](https://github.com/microsoft/DeepSpeed)
```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .
ds_report # check the deespeed status
```
- [Tips] For single-node docker environments, we need to set up ports for your ssh. e.g., root@127.0.0.1 with port 7110
```
>>> vim ~/.ssh/config
Host 127.0.0.1
    Hostname 127.0.0.1
    Port 7110
    User root
```
- [Tips] For multi-node docker environments, generate ssh keys and copy the public key to all nodes (in `~/.ssh/`)
```
>>> ssh-keygen -t rsa -C "xxx@xxx.com"
```

## Quick Start
We provide many models which are trained to perform different tasks. You can load these models by AutoLoader to make prediction. See more in `FlagAI/quickstart`.
## Load model and tokenizer
We provide the AutoLoad class to load the model and tokenizer quickly, for example:
```python
from flagai.auto_model.auto_loader import AutoLoader

auto_loader = AutoLoader(
    task_name="title-generation",
    model_name="BERT-base-en"
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
This example is for the `title_generation` task, and you can also model other tasks by modifying the `task_name`.
Then you can use the model and tokenizer to fine-tune or test.

## Predictor
We provide the `Predictor` class to predict for different tasks, for example:

```python
from flagai.model.predictor.predictor import Predictor
predictor = Predictor(model, tokenizer)
test_data = [
    "Four minutes after the red card, Emerson Royal nodded a corner into the path of the unmarked Kane at the far post, who nudged the ball in for his 12th goal in 17 North London derby appearances. Arteta's misery was compounded two minutes after half-time when Kane held the ball up in front of goal and teed up Son to smash a shot beyond a crowd of defenders to make it 3-0.The goal moved the South Korea talisman a goal behind Premier League top scorer Mohamed Salah on 21 for the season, and he looked perturbed when he was hauled off with 18 minutes remaining, receiving words of consolation from Pierre-Emile Hojbjerg.Once his frustrations have eased, Son and Spurs will look ahead to two final games in which they only need a point more than Arsenal to finish fourth.",
]

for text in test_data:
    print(
        predictor.predict_generate_beamsearch(text,
                                              out_max_length=50,
                                              beam_size=3))
```
This example is for the `seq2seq` task, where we can get `beam-search` results by calling the `predict_generate_beamsearch` function. In addition, we also support prediction for tasks such as `NER` and `title generate`.


## NER

```python
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

task_name = "ner"
model_name = "RoBERTa-base-ch"
target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
maxlen = 256

auto_loader = AutoLoader(task_name,
                         model_name=model_name,
                         load_pretrain_params=True,
                         class_num=len(target))

model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

predictor = Predictor(model, tokenizer)

test_data = [
    "6月15日，河南省文物考古研究所曹操高陵文物队公开发表声明承认：“从来没有说过出土的珠子是墓主人的",
    "4月8日，北京冬奥会、冬残奥会总结表彰大会在人民大会堂隆重举行。习近平总书记出席大会并发表重要讲话。在讲话中，总书记充分肯定了北京冬奥会、冬残奥会取得的优异成绩，全面回顾了7年筹办备赛的不凡历程，深入总结了筹备举办北京冬奥会、冬残奥会的宝贵经验，深刻阐释了北京冬奥精神，对运用好冬奥遗产推动高质量发展提出明确要求。",
    "当地时间8日，欧盟委员会表示，欧盟各成员国政府现已冻结共计约300亿欧元与俄罗斯寡头及其他被制裁的俄方人员有关的资产。",
    "这一盘口状态下英国必发公司亚洲盘交易数据显示博洛尼亚热。而从欧赔投注看，也是主队热。巴勒莫两连败，",
]

for t in test_data:
    entities = predictor.predict_ner(t, target, maxlen=maxlen)
    result = {}
    for e in entities:
        if e[2] not in result:
            result[e[2]] = [t[e[0]:e[1] + 1]]
        else:
            result[e[2]].append(t[e[0]:e[1] + 1])
    print(f"result is {result}")
```

## Semantic Matching

```python
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

maxlen = 256

auto_loader = AutoLoader("semantic-matching",
                         model_name="RoBERTa-base-ch",
                         load_pretrain_params=True,
                         class_num=2)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

predictor = Predictor(model, tokenizer)

test_data = [["后悔了吗", "你有没有后悔"], ["打开自动横屏", "开启移动数据"],
             ["我觉得你很聪明", "你聪明我是这么觉得"]]

for text_pair in test_data:
    print(predictor.predict_cls_classifier(text_pair))

```

## Pre-trained Models and examples

* [Text_image_matching with AltCLIP](/examples/AltCLIP/README.md)
* [Text-to-image generation with AltDiffusion](/examples/AltDiffusion/README.md)
* [Blank_Filling_QA with GLM ](/docs/TUTORIAL_11_GLM_BLANK_FILLING_QA.md)
* [Title Generation with GLM ](/docs/TUTORIAL_12_GLM_EXAMPLE_TITLE_GENERATION.md)
* [Poetry generation with GLM-large-ch](docs/TUTORIAL_13_GLM_EXAMPLE_PEOTRY_GENERATION.md)
* [Using huggingface's t5-11b & tricks ](docs/TUTORIAL_14_HUGGINGFACE_T5.md)
* [Title Generation with RoBerta-WWM](/docs/TUTORIAL_15_BERT_EXAMPLE_TITLE_GENERATION.md)
* [Semantic Matching with RoBerta-WWM](/docs/TUTORIAL_16_BERT_EXAMPLE_SEMANTIC_MATCHING.md)
* [NER with RoBerta-WWM](/docs/TUTORIAL_17_BERT_EXAMPLE_NER.md)
* [Writing with GPT-2](/docs/TUTORIAL_18_GPT2_WRITING.md)
* [Title generation with T5](/docs/TUTORIAL_19_T5_EXAMPLE_TITLE_GENERATION.md)
* [Example of OPT](/examples/opt/README.md)

[//]: # (* [Supported tasks]&#40;/docs/TUTORIAL_20_SUPPORTED_TASKS.md&#41;)


This session explains how the base NLP classes work, how you can load pre-trained models to tag your
text, how you can embed your text with different word or document embeddings, and how you can train your own
language models, sequence labeling models, and text classification models. Let us know if anything is unclear. See more in `FlagAI/examples`.



## Tutorials
We provide a set of quick tutorials to get you started with the library:
* [Tutorial 1: How to construct and use Tokenizer](/docs/TUTORIAL_1_TOKENIZER.md)
* [Tutorial 2: Dataset Preprocessing Pipeline](/docs/TUTORIAL_2_DATASET.md)
* [Tutorial 3: Major Function of Model Module](/docs/TUTORIAL_3_MODEL.md)
* [Tutorial 4: Customize trainer for model and data-parallel training](/docs/TUTORIAL_4_TRAINER.md)
* [Tutorial 5: Simplify model and tokenizer Initialization by Using Autoloader](/docs/TUTORIAL_5_INSTRUCTIONS_FOR_AutoLoader.md)
* [Tutorial 6: Use off-the-shelf inference Algorithms with Predictor](/docs/TUTORIAL_6_INSTRUCTIONS_FOR_PREDICTOR.md)
* [Tutorial 7: Use FlagAI prompt-learning tool-kit to improve performance on SuperGLUE](/docs/TUTORIAL_7_PROMPT_LERANING.md)
* [Tutorial 8: Setup environment for training models with multi-machine](/docs/TUTORIAL_8_ENVIRONMENT_SETUP.md)
* [Tutorial 9: Text generation with encoder/decoder/encoder-decoder models](/docs/TUTORIAL_9_SEQ2SEQ_METHOD.md)
* [Tutorial 10: How to transform a customized model into a megatron-LM-style parallel model](/docs/TUTORIAL_10_MEGATRON.md)

## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/FlagAI-Open/FlagAI/issues) for specific tasks.

## Contact us

<img src="./flagai_wechat.png" width = "200" height = "200"  align=center />

## [License](/LICENSE)
The majority of FlagAI is licensed under the [Apache 2.0 license](LICENSE), however portions of the project are available under separate license terms:

* Megatron-LM is licensed under the [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)
* GLM is licensed under the [MIT license](https://github.com/THUDM/GLM/blob/main/LICENSE)
* AltDiffusion is licensed under the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
