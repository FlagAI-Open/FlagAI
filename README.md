![FlagAI](logo.png)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6052/badge)](https://bestpractices.coreinfrastructure.org/projects/6052)
[![Python package](https://github.com/BAAI-Open/FlagAI/actions/workflows/python-package.yml/badge.svg)](https://github.com/BAAI-Open/FlagAI/actions/workflows/python-package.yml)
[简体中文](README_zh.md)

--------------------------------------------------------------------------------


FlagAI (Fast LArge-scale General AI models) is an fast, easy-to-use and extensible toolkit for large-scale model. Our goal is to support training, fine-tuning, and deployment of large-scale models on various downstream tasks with multi-modality. Currently, we are focusing on NLP models and tasks. In near futher, we will support for other modalities.

* Now it supports **WuDao GLM** with a maximum of 10 billion parameters (see [GLM details](docs/GLM.md)). It also supports **BERT**, **RoBERTa**, **GPT2**, **T5**, and models from Huggingface Transformers.

* It provides APIs to quickly download and use those pre-trained models on a given text, fine-tune them on widely-used datasets collected from [SuperGLUE](https://super.gluebenchmark.com/) and [CLUE](https://github.com/CLUEbenchmark/CLUE) benchmarks, and then share them with the community on our model hub. It also provides [prompt-learning]() toolkit for few shot tasks.   

* These models can be applied to (Chinese/English) Text, for tasks like text classification, information extraction, question answering, summarization, and text generation.

* FlagAI is backed by the three most popular data/model parallel libraries — [PyTorch](https://pytorch.org/)/[Deepspeed](https://www.deepspeed.ai/)/[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — with seamless integration between them. Users can parallel their training/testing process with less than ten lines of code.


The code is partially based on [Transformers](https://github.com/huggingface/transformers) and [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples).


<!-- toc -->

- [Requirements and Installation](#requirements-and-installation)
- [Quick Started](#quick-start)
    - [Load model and tokenizer](#load-model-and-tokenizer)
    - [Predictor](#predictor)
    - [NER task](#ner-task)
    - [Title generation task](#title-generation-task)
    - [Semantic matching task](#semantic-matching-task)
- [Pretrained Models and examples](#pretrained-models-and-examples)
- [Tutorials](#tutorials)
- [Learn More About FlagAI](#learn-more-about-FlagAI)
- [Contributing](#contributing)
- [License](#license)

<!-- tocstop -->
## Requirements and Installation
* PyTorch version >= 1.8.0
* Python version >= 3.8
* For training/testing models on GPUs, you'll also need install CUDA and NCCL

To install FlagAI with pip:
```shell
pip install -U flagai
```

- [Optional]To install FlagAI and develop locally:

```shell
git clone https://github.com/BAAI-Open/FlagAI.git
python setup.py install
```

- [Optional] For faster training install NVIDIA's [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- [Optional] For ZeRO optimizers install [DEEPSPEED](https://github.com/microsoft/DeepSpeed)
```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .
ds_report # check the deespeed status
```
- [Tips] For single-node docker enviroments, we need to setup ports for your ssh. e.g., root@127.0.0.1 with port 7110
```
>>> vim ~/.ssh/config
Host 127.0.0.1
    Hostname 127.0.0.1
    Port 7110
    User root
```
- [Tips] For multi-node docker enviroments, generate ssh keys and copy the public key to all nodes (in `~/.ssh/`)
```
>>> ssh-keygen -t rsa -C "xxx@xxx.com"
```

## Quick Start
We provide many models which are trained to perform different tasks. You can load these models by AutoLoader to make prediction.
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
Then you can use the model and tokenizer to finetune or test.

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

## Pretrained Models and examples

* [Blank_Filling_QA with GLM ](/docs/TUTORIAL_11_GLM_BLANK_FILLING_QA.md)
* [Title Generation with GLM ](/docs/TUTORIAL_12_GLM_EXAMPLE_TITLE_GENERATION.md)
* [Poetry generation with GLM-large-ch](docs/TUTORIAL_13_GLM_EXAMPLE_PEOTRY_GENERATION.md)
* [Using huggingface's t5-3b & tricks ](docs/TUTORIAL_14_HUGGINGFACE_T5.md)
* [Title Generation with RoBerta-WWM](/docs/TUTORIAL_15_BERT_EXAMPLE_TITLE_GENERATION.md)
* [Semantic Matching with RoBerta-WWM](/docs/TUTORIAL_16_BERT_EXAMPLE_SEMANTIC_MATCHING.md)
* [NER with RoBerta-WWM](/docs/TUTORIAL_17_BERT_EXAMPLE_NER.md)
* [Writing with GPT-2](/docs/TUTORIAL_18_GPT2_WRITING.md)
* [Title generation with T5](/docs/TUTORIAL_19_T5_EXAMPLE_TITLE_GENERATION.md)
* [Supported tasks](/docs/TUTORIAL_20_SUPPORTED_TASKS.md)


This session explains how the base NLP classes work, how you can load pre-trained models to tag your
text, how you can embed your text with different word or document embeddings, and how you can train your own
language models, sequence labeling models, and text classification models. Let us know if anything is unclear.



## Tutorials
We provide a set of quick tutorials to get you started with the library:

[//]: # (* [Tutorial 1: Background: Transformers]&#40;docs/TUTORIAL_1_BASICS.md&#41;)
* [Tutorial 1: How to construct and use Tokenizer](docs/TUTORIAL_1_TOKENIZER.md)
* [Tutorial 2: Dataset Preprocessing Pipeline](docs/TUTORIAL_2_DATASET.md)
* [Tutorial 3: Major Function of Model Module](docs/TUTORIAL_3_MODEL.md)
* [Tutorial 4: Customize trainer for model and data-parallel training](docs/TUTORIAL_4_TRAINER.md)
* [Tutorial 5: Simplify model and tokenizer Initialization by Using Autoloader](docs/TUTORIAL_5_INSTRUCTIONS_FOR_AutoLoader.md)
* [Tutorial 6: Use off-the-shelf inference Algorithms with Predictor](docs/TUTORIAL_6_INSTRUCTIONS_FOR_PREDICTOR.md)

* [Tutorial 7: Use FlagAI prompt-learning tool-kit to improve performance on SuperGLUE](docs/TUTORIAL_7_PROMPT_LERANING.md)
* [Tutorial 8: Setup environment for training models with multi-machine](docs/TUTORIAL_8_ENVIRONMENT_SETUP.md)
* [Tutorial 9: Text generation with encoder/decoder/encoder-decoder models](docs/TUTORIAL_9_SEQ2SEQ_METHOD.md)
* [Tutorial 11: How to transform a customized model into a megatron-LM-style parallel model](docs/TUTORIAL_10_MEGATRON.md)
* 
## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/BAAI-WuDao/Sailing/issues) for specific tasks.


## [License](/LICENSE)
```
Copyright [2022] [BAAI]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
