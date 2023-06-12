![FlagAI](logo.png)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6052/badge)](https://bestpractices.coreinfrastructure.org/projects/6052)
[![Python application](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml/badge.svg)](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/FlagAI-Open/FlagAI?include_prereleases&style=social)

[简体中文](README_zh.md)

--------------------------------------------------------------------------------


FlagAI (Fast LArge-scale General AI models) is a fast, easy-to-use and extensible toolkit for large-scale model. Our goal is to support training, fine-tuning, and deployment of large-scale models on various downstream tasks with multi-modality.



## Why should I use FlagAI?


1. **Quickly Download Models via API**

    FlagAI provides an API that allows you to quickly download pre-trained models and fine-tune them on a wide range of datasets collected from [SuperGLUE](https://super.gluebenchmark.com/) and [CLUE](https://github.com/CLUEbenchmark/CLUE) benchmarks for both Chinese and English text.

    FlagAI now supports over 30 mainstream models, including Language Model [**Aquila**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila), multilingual text and image representation model [**AltCLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP), text-to-image generation model [**AltDiffusion**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/BAAI/bilingual_stable_diffusion), [**WuDao GLM**](/docs/GLM.md) (with a maximum of 10 billion parameters), [**EVA-CLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP), **OPT**, **BERT**, **RoBERTa**, **GPT2**, **T5**, **ALM**, and models from **Huggingface Transformers**, etc.
    

2. **Parallel train with fewer than 10 lines of code**

	Backed by the four most popular data/model parallel libraries -- [PyTorch](https://pytorch.org/), [Deepspeed](https://www.deepspeed.ai/), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [BMTrain](https://github.com/OpenBMB/BMTrain) -- FlagAI allows for seamless integration between them, enabling users to parallel their training/testing process with fewer than ten lines of code.


3. **Conveniently use the few-shot learning toolkits**
   
    FlagAI also provides [prompt-learning](/docs/TUTORIAL_7_PROMPT_LEARNING.md) toolkit for few-shot tasks.

4. **Particularly good at Chinese tasks**

    These models can be applied to (Chinese/English) Text, for tasks like text classification, information extraction, question answering, summarization, and text generation, with a particular focus on Chinese tasks.


## Toolkits and Pre-trained Models 

> The code is partially based on [GLM](https://github.com/THUDM/GLM), [Transformers](https://github.com/huggingface/transformers)，[timm](https://github.com/rwightman/pytorch-image-models) and [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM).


### Toolkits


| Name       | Description       | Examples            |
|:-------------- |:---------- |:------------------------------------------------------ |
| 	`GLM_custom_pvp` | Customizing PET templates   | [README.md](http:///examples/glm_custom_pvp/README.md) |
| `GLM_ptuning`    | p-tuning tool | ——                                                     |
| `BMInf-generate` | Accelerating generation | [README.md](http:///examples/bminf_generate/README.md) | 


### Pre-trained Models 


|   Model          |  Task    | Train | Finetune | Inference/Generate | Examples       |                                                         
| :---------------- | :------- | :-- |:-- | :-- | :--------------------------------------------- |
| Aquila      | Natural Language Processing  | ✅  | ✅  | ✅  | [README.md](examples/Aquila/README.md) 
| ALM          | Arabic Text Generation  |  ✅  | ❌  | ✅  | [README.md](/examples/ALM/README.md)  |                         
| AltCLIP       | Image-Text Matching  | ✅  | ✅  | ✅  | [README.md](/examples/AltCLIP/README.md)   |  
| AltCLIP-m18      | Image-Text Matching  | ✅  | ✅  | ✅  | [README.md](examples/AltCLIP-m18/README.md)   |                             
| AltDiffusion    | Text-to-Image Generation    | ❌  | ❌  | ✅  | [README.md](/examples/AltDiffusion/README.md)    |
| AltDiffusion-m18    | Text-to-Image Generation,supporting 18 languages    | ❌  | ❌  | ✅  |[README.md](/examples/AltDiffusion-m18/README.md)   |
| BERT-title-generation-english     | English Title Generation | ✅  | ❌  | ✅  | [README.md](/examples/bert_title_generation_english/README.md) |
| CLIP           | Image-Text Matching    | ✅  | ❌  | ✅  | ——   |                                                                 
| CPM3-finetune       | Text Continuation   | ❌  | ✅  | ❌  | ——    |                                                                
| CPM3-generate    | Text Continuation  | ❌  | ❌  | ✅  | ——   |                                                                 
| CPM3_pretrain    | Text Continuation  | ✅  | ❌  | ❌  | ——        |
| CPM_1     | Text Continuation   | ❌  | ❌  | ✅  | [README.md](/examples/cpm_1/README.md)      |
| EVA-CLIP                          | Image-Text Matching  | ✅  | ✅  | ✅  | [README.md](/examples/EVA_CLIP/README.md)                             |
| Galactica       | Text Continuation    | ❌  | ❌  | ✅  | ——      |                                                              
| GLM-large-ch-blank-filling        | Blank Filling     | ❌  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_11_GLM_BLANK_FILLING_QA.md)               |
| GLM-large-ch-poetry-generation    | Poetry Generation     | ✅  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_13_GLM_EXAMPLE_PEOTRY_GENERATION.md)       |
| GLM-large-ch-title-generation     | Title Generation   | ✅  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_12_GLM_EXAMPLE_TITLE_GENERATION.md)        |
| GLM-pretrain         | Pre-Train    | ✅  | ❌  | ❌  | ——   |                                                                 
| GLM-seq2seq        | Generation    | ✅  | ❌  | ✅  | ——     |                                                               
| GLM-superglue      | Classification  | ✅  | ❌  | ❌  | ——     |                                                               
| GPT-2-text-writting      | Text Continuation   | ❌  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_18_GPT2_WRITING.md)        |
| GPT2-text-writting                | Text Continuation   | ❌  | ❌  | ✅  | —— |                                                                   
| GPT2-title-generation             | Title Generation   | ❌  | ❌  | ✅  | ——  |                                                                  
| OPT                               | Text Continuation   | ❌  | ❌  | ✅  | [README.md](/examples/opt/README.md) |                                  
| RoBERTa-base-ch-ner               | Named Entity Recognition| ✅  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_17_BERT_EXAMPLE_NER.md)     |
| RoBERTa-base-ch-semantic-matching |Semantic Similarity Matching | ✅  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_16_BERT_EXAMPLE_SEMANTIC_MATCHING.md)      |
| RoBERTa-base-ch-title-generation  | Title Generation     | ✅  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_15_BERT_EXAMPLE_TITLE_GENERATION.md)       |
| RoBERTa-faq      |   Question-Answer   | ❌  | ❌  | ✅  | [README.md](/examples/roberta_faq/README.md) |         
| Swinv1                            | Image Classification | ✅  | ❌  | ✅  | ——  |                                                                  
| Swinv2                            | Image Classification   | ✅  | ❌  | ✅  | ——     |                                                               
| T5-huggingface-11b                | Train   | ✅  | ❌  | ❌  | [TUTORIAL](/docs/TUTORIAL_14_HUGGINGFACE_T5.md)                      |
| T5-title-generation               | Title Generation     | ❌  | ❌  | ✅  | [TUTORIAL](/docs/TUTORIAL_19_T5_EXAMPLE_TITLE_GENERATION.md)                |
| T5-flagai-11b                     | Pre-Train  | ✅  | ❌  | ❌  | ——    |                                                                
| ViT-cifar100                      |  Pre-Train  | ✅  | ❌  | ❌  | —— |


> * More excamples in  [./examples](https://github.com/FlagAI-Open/FlagAI/tree/master/examples) 
> * More tutorials in [./docs](https://github.com/FlagAI-Open/FlagAI/tree/master/doc) 




## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/FlagAI-Open/FlagAI/issues) for specific tasks.


## Contact us

Welcome to raise your questions or feature requests on [GitHub Issues](https://github.com/FlagAI-Open/FlagAI/issues) , and share your experience on the  [Discussions](https://github.com/FlagAI-Open/FlagAI/discussions) board.

* Official email: open.platform@baai.ac.cn.
* Zhihu: [FlagAI](https://www.zhihu.com/people/95-22-20-18)
* Scan the qrcode to join the WeChat group for communication:

<img src="./wechat-qrcode.jpg" width = "200" height = "200"  align=center />


## Quick Start
We provide many models which are trained to perform different tasks. You can load these models by AutoLoader to make prediction. See more in `FlagAI/quickstart`.

### Requirements and Installation
* Python version >= 3.8
* PyTorch version >= 1.8.0
* [Optional] For training/testing models on GPUs, you'll also need to install CUDA and NCCL

- To install FlagAI with pip:
```shell
pip install -U flagai
```

- [Optional] To install FlagAI and develop locally:

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
- [Optional] For ZeRO optimizers, install [DEEPSPEED](https://github.com/microsoft/DeepSpeed) (>= 0.7.7)
```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .
ds_report # check the deespeed status
```
- [Optional] For BMTrain training, install [BMTrain](https://github.com/OpenBMB/BMTrain)
```
git clone https://github.com/OpenBMB/BMTrain
cd BMTrain
python setup.py install
```
- [Optional] For BMInf low-resource inference, install [BMInf](https://github.com/OpenBMB/BMInf)
```
pip install bminf

```
- [Optional] For Flash Attention, install [Flash-attention](https://github.com/HazyResearch/flash-attention) (>=1.0.2)
```
pip install flash-attn
```

- [Tips] For single-node docker environments, we need to set up ports for your ssh. e.g., root@127.0.0.1 with port 711
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


### Load model and tokenizer
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

### Examples

#### 1. Predictor 

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


#### 2. NER 

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

#### 3. Semantic Matching example

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



## LICENSE

The majority of FlagAI is licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0), however portions of the project are available under separate license terms:

* Megatron-LM is licensed under the [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)
* GLM is licensed under the [MIT license](https://github.com/THUDM/GLM/blob/main/LICENSE)
* AltDiffusion is licensed under the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)



## News
- [9 June 2023] release v1.7.0, Support Aquila [#324](https://github.com/FlagAI-Open/FlagAI/pull/324);
- [31 Mar 2023] release v1.6.3, Support AltCLIP-m18 [#303](https://github.com/FlagAI-Open/FlagAI/pull/303) and AltDiffusion-m18 [#302](https://github.com/FlagAI-Open/FlagAI/pull/302); 
- [17 Mar 2023] release v1.6.2, Support application of new optimizers [#266](https://github.com/FlagAI-Open/FlagAI/pull/266), and added a new gpt model name 'GPT2-base-en' for English; 
- [2 Mar 2023] release v1.6.1, Support Galactica model [#234](https://github.com/FlagAI-Open/FlagAI/pull/234); BMInf, a low-resource inference package [#238](https://github.com/FlagAI-Open/FlagAI/pull/238), and examples for p-tuning [#227](https://github.com/FlagAI-Open/FlagAI/pull/238)
- [12 Jan 2023] release v1.6.0, support a new parallel lib called [**BMTrain**](https://github.com/OpenBMB/BMTrain) and integate [**Flash Attention**](https://github.com/HazyResearch/flash-attention) to speedup training of BERT and ViT models, examples in [FlashAttentionBERT](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/bert_title_generation_english/train_flash_atten.py) and [FlashAttentionViT](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/vit_cifar100/train_single_gpu_flash_atten.py). Also add the contrastive search based text generation method [**SimCTG**](https://github.com/yxuansu/SimCTG) and DreamBooth finetuning based on AltDiffusion, examples in [AltDiffusionNaruto](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/AltDiffusion/dreambooth.py). 
- [28 Nov 2022] release v1.5.0, support 1.1B [**EVA-CLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP) and [ALM: A large Arabic Language Model based on GLM], examples in [**ALM**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/ALM)
- [10 Nov 2022] release v1.4.0, support [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v1), examples in [**AltCLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) and [**AltDiffusion**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion)
- [29 Aug 2022] release v1.3.0, Added CLIP module and redesigned tokenizer APIs in [#81](https://github.com/FlagAI-Open/FlagAI/pull/81)
- [21 Jul 2022] release v1.2.0, ViTs are supported in [#71](https://github.com/FlagAI-Open/FlagAI/pull/71)
- [29 Jun 2022] release v1.1.0, support OPTs downloading and inference/fine-tuning [#63](https://github.com/FlagAI-Open/FlagAI/pull/63)
- [17 May 2022] made our first contribution in [#1](https://github.com/FlagAI-Open/FlagAI/pull/1)

## Platforms supported

<div  align="center">    
<img src="./examples/Aquila/img/merged_platform.jpg" height = "100" align=center />
</div>



## Misc

### &#8627; Stargazers, thank you for your support!
[![Stargazers repo roster for @FlagAI-Open/FlagAI](https://reporoster.com/stars/FlagAI-Open/FlagAI)](https://github.com/FlagAI-Open/FlagAI/stargazers)

### &#8627; Forkers, thank you for your support!
[![Forkers repo roster for @FlagAI-Open/FlagAI](https://reporoster.com/forks/FlagAI-Open/FlagAI)](https://github.com/FlagAI-Open/FlagAI/network/members)

### &#8627; Star History
<div align="center">

![Star History Chart](https://api.star-history.com/svg?repos=FlagAI-Open/FlagAI&type=Date)]

</div>
