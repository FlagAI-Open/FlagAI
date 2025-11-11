[<img src="flagopen.jpeg">](https://flagopen.baai.ac.cn/)
![FlagAI](logo.png)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6052/badge)](https://bestpractices.coreinfrastructure.org/projects/6052)
[![Python application](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml/badge.svg)](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/FlagAI-Open/FlagAI?include_prereleases&style=social)

[English](README.md)

--------------------------------------------------------------------------------


**FlagAI飞智**是一个快速、易于使用和可扩展的大模型工具包。 我们的目标是支持在多模态的各种下游任务上训练、微调和部署大规模模型。
<br><br>

## 为什么你需要 FlagAI?

1. **可通过 API 快速下载模型**
      
    提供 API 方便你快速下载模型，并在给定（中/英文）文本上使用这些预训练模型，在从[SuperGLUE](https://super.gluebenchmark.com/)和[CLUE](https://github.com/CLUEbenchmark/CLUE) benchmarks收集的广泛使用的数据集上对它们进行微调。
     
      FlagAI 现已支持 30+ 主流模型，包括语言模型[**Aquila**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila), 多模态模型 [**AltCLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) 、文生图模型 [**AltDiffusion**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/BAAI/bilingual_stable_diffusion)、最高百亿参数的 **[悟道GLM](/doc_zh/GLM.md)**，[**EVA-CLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP)、**[Galactica](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/galactica)**、**OPT**、**BERT**、**RoBERTa**、**GPT2**、**T5**、**ALM**、**Huggingface Transformers** 等。
      
2.  **仅用十行代码即可进行并行训练**

    飞智由四个最流行的数据/模型并行库（[PyTorch](https://pytorch.org/)/[Deepspeed](https://www.deepspeed.ai/)/[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)/[BMTrain](https://github.com/OpenBMB/BMTrain)）提供支持，它们之间实现了无缝集成。 你可以用不到十行代码来并行你的训练/测试过程。
   
3.  **提供提示学习工具包**

    FlagAI 提供了提示学习（[prompt-learning](https://github.com/FlagAI-Open/FlagAI/blob/master/docs/TUTORIAL_7_PROMPT_LEARNING.md)）的工具包，用于少样本学习(few-shot learning)任务。
   
4.  **尤其擅长中文任务**

    FlagAI 目前支持的模型可以应用于文本分类、信息提取、问答、摘要、文本生成等任务，尤其擅长中文任务。



## 工具包及已支持的模型

> 本项目的部分代码基于 [GLM](https://github.com/THUDM/GLM)，[Transformers](https://github.com/huggingface/transformers)，[timm](https://github.com/rwightman/pytorch-image-models) 和 [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM).


### 工具

| 工具名称           | 描述         | 样例                |
|:-------------- |:---------- |:------------------------------------------------------ |
| 	`GLM_custom_pvp` | 自定义 PET 模板   | [README.md](./examples/glm_custom_pvp/README.md) |
| `GLM_ptuning`    | p-tuning 工具 | ——                                                     |
| `BMInf-generate` | 推理加速    | [README.md](./examples/bminf_generate/README.md) |

### 模型

|    模型名称            | 任务      | 训练 | 微调 | 推理 | 样例           |                                                         
| :---------------- | :------- | :-- |:-- | :-- | :--------------------------------------------- |
| Aquila      | 自然语言处理  | ✅  | ✅  | ✅  | [README.md](examples/Aquila/README.md) 
| ALM          | 阿拉伯语文本生成   |  ✅  | ❌  | ✅  | [README.md](/examples/ALM/README.md)  |                         
| AltCLIP       | 文图匹配 | ✅  | ✅  | ✅  | [README.md](/examples/AltCLIP/README.md)   |  
| AltCLIP-m18      | 文图匹配  | ✅  | ✅  | ✅  | [README.md](examples/AltCLIP-m18/README.md)   |                             
| AltDiffusion    | 文生图  | ❌  | ❌  | ✅  | [README.md](/examples/AltDiffusion/README.md)    |
| AltDiffusion-m18    | 文生图，支持 18 种语言   | ❌  | ❌  | ✅  | [README.md](/examples/AltDiffusion-m18/README.md)   |
| BERT-title-generation-english     | 英文标题生成  | ✅  | ❌  | ✅  | [README.md](/examples/bert_title_generation_english/README.md) |
| CLIP           | 图文匹配    | ✅  | ❌  | ✅  | ——   |                                                                 
| CPM3-finetune       | 文本续写    | ❌  | ✅  | ❌  | ——    |                                                                
| CPM3-generate    | 文本续写    | ❌  | ❌  | ✅  | ——   |                                                                 
| CPM3_pretrain    | 文本续写    | ✅  | ❌  | ❌  | ——        |
| CPM_1     | 文本续写    | ❌  | ❌  | ✅  | [README.md](/examples/cpm_1/README.md)      |
| EVA-CLIP                          | 图文匹配    | ✅  | ✅  | ✅  | [README.md](/examples/EVA_CLIP/README.md)                             |
| Galactica       | 文本续写    | ❌  | ❌  | ✅  | ——      |                                                              
| GLM-large-ch-blank-filling        | 完形填空问答  | ❌  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_11_GLM_BLANK_FILLING_QA.md)               |
| GLM-large-ch-poetry-generation    | 诗歌生成    | ✅  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_13_GLM_EXAMPLE_PEOTRY_GENERATION.md)       |
| GLM-large-ch-title-generation     | 标题生成    | ✅  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_12_GLM_EXAMPLE_TITLE_GENERATION.md)        |
| GLM-pretrain         | 预训练     | ✅  | ❌  | ❌  | ——   |                                                                 
| GLM-seq2seq        | 生成任务    | ✅  | ❌  | ✅  | ——     |                                                               
| GLM-superglue      | 判别任务    | ✅  | ❌  | ❌  | ——     |                                                               
| GPT-2-text-writting      | 文本续写    | ❌  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_18_GPT2_WRITING.md)        |
| GPT2-text-writting                | 文本续写    | ❌  | ❌  | ✅  | —— |                                                                   
| GPT2-title-generation             | 标题生成    | ❌  | ❌  | ✅  | ——  |                                                                  
| OPT                               | 文本续写    | ❌  | ❌  | ✅  | [README.md](/examples/opt/README.md) |                                  
| RoBERTa-base-ch-ner               | 命名实体识别  | ✅  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_17_BERT_EXAMPLE_NER.md)     |
| RoBERTa-base-ch-semantic-matching | 语义相似度匹配 | ✅  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_16_BERT_EXAMPLE_SEMANTIC_MATCHING.md)      |
| RoBERTa-base-ch-title-generation  | 标题生成    | ✅  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_15_BERT_EXAMPLE_TITLE_GENERATION.md)       |
| RoBERTa-faq      | 问答      | ❌  | ❌  | ✅  | [README.md](/examples/roberta_faq/README.md) |         
| Swinv1                            | 图片分类    | ✅  | ❌  | ✅  | ——  |                                                                  
| Swinv2                            | 图片分类    | ✅  | ❌  | ✅  | ——     |                                                               
| T5-huggingface-11b                | 训练      | ✅  | ❌  | ❌  | [TUTORIAL](/doc_zh/TUTORIAL_14_HUGGINGFACE_T5.md)                      |
| T5-title-generation               | 标题生成    | ❌  | ❌  | ✅  | [TUTORIAL](/doc_zh/TUTORIAL_19_T5_EXAMPLE_TITLE_GENERATION.md)                |
| T5-flagai-11b                     | 预训练     | ✅  | ❌  | ❌  | ——    |                                                                
| ViT-cifar100                      | 预训练     | ✅  | ❌  | ❌  | —— |


> 更多样例见 [./examples](https://github.com/FlagAI-Open/FlagAI/tree/master/examples) 目录，更多中文教程见 [./docs_zh](https://github.com/FlagAI-Open/FlagAI/tree/master/doc_zh) 目录。


## 贡献代码

感谢您对贡献的兴趣！请先阅读 [贡献者指南](CONTRIBUTING.md)，然后从 [未解决的问题](https://github.com/FlagAI-Open/FlagAI/issues) 寻找你感兴趣的任务开启贡献之旅！

## 联系我们

欢迎在 [GitHub Issues](https://github.com/FlagAI-Open/FlagAI/issues) 中提出你的问题，或在 [Discussions ](https://github.com/FlagAI-Open/FlagAI/discussions) 板块交流使用经验。

* 官方邮箱：open.platform@baai.ac.cn。
* 知乎：[FlagAI飞智](https://www.zhihu.com/people/95-22-20-18)
* 扫码添加小助手加入**微信交流群**：

<img src="./wechat-qrcode.jpg" width = "200" height = "200"  align=center />



## Quick Start

### 安装环境

* Python 版本 >= 3.8
* PyTorch 版本 >= 2.9.0
* [可选] 使用GPUs进行训练和测试, 你需要安装CUDA 和 NCCL

- 通过`pip`安装:
```shell
pip install -U flagai
```

- [可选] 下载源码安装:

```shell
git clone https://github.com/FlagAI-Open/FlagAI.git
python setup.py install
```

- [可选] 开启训练加速需要安装 NVIDIA的 [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- [可选] 使用 ZeRO 优化器，需要安装 [DEEPSPEED](https://github.com/microsoft/DeepSpeed) (>= 0.7.7)
```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .
ds_report # 检查deepspeed的状态
```
- [可选] 开启BMTrain训练，需要安装 [BMTrain](https://github.com/OpenBMB/BMTrain) ((>= 0.2.2))
```
git clone https://github.com/OpenBMB/BMTrain
cd BMTrain
python setup.py install 
```

- [可选] 开启BMInf低资源推理, 需要安装[BMInf](https://github.com/OpenBMB/BMInf)
```
pip install bminf

```
- [可选] 对于FlashAttention, 需要安装[Flash-attention](https://github.com/HazyResearch/flash-attention) （>=3.0.0）
```
pip install flash-attn --no-build-isolation
```
注意：Flash Attention 3.0 需要 CUDA 11.7+ 和 PyTorch 2.9.0+。对于较旧的 GPU（Turing 架构），请使用 flash-attn 1.x 版本。

- [可选] 镜像构建，请参照 [Dockerfile](https://github.com/FlagAI-Open/FlagAI/blob/master/Dockerfile)
- [提示] 单节点docker环境下，运行多卡数据并行需要设置host。 例如，docker节点 root@127.0.0.1，其端口 7110。
```
>>> vim ~/.ssh/config
Host 127.0.0.1
    Hostname 127.0.0.1
    Port 7110
    User root
```
- [提示] 多节点环境， 需要生成 ssh keys 并拷贝公钥到所有节点 (in `~/.ssh/`)
```
>>> ssh-keygen -t rsa -C "xxx@xxx.com"
```

### 加载模型和分词器
我们提供 `AutoLoad` 类来快速加载模型和分词器，例如：

```python
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(
      task_name="title-generation",
      model_name="RoBERTa-base-ch"  
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```

这个例子是针对`title-generation`(文本摘要）任务的，你也可以通过修改`task_name`来为其他任务建模。 然后您可以使用模型和标记器进行微调或测试。

### 使用预测器
我们提供 `Predictor` 类来预测不同的任务，例如：

```python
from flagai.model.predictor.predictor import Predictor
predictor = Predictor(model, tokenizer)
test_data = [
    "本文总结了十个可穿戴产品的设计原则而这些原则同样也是笔者认为是这个行业最吸引人的地方1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人",
    "2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献",
    "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"
]
for text in test_data:
    print(
        predictor.predict_generate_beamsearch(text,
                                              out_max_length=50,
                                              beam_size=3))
```

这个例子是针对 `seq2seq` 任务的，我们可以通过调用`predict_generate_beamsearch`函数得到`beam-search`结果。此外，我们还支持`NER`和`title generate`等任务的预测。


### 命名实体识别任务示例

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


### 语义相似度匹配任务示例

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


## 动态
- [9 June 2023] 支持 v1.7.0版本, 增加Aquila [#324](https://github.com/FlagAI-Open/FlagAI/pull/324);
- [31 Mar 2023] 支持v1.6.3版本, 增加AltCLIP-m18模型 [#303](https://github.com/FlagAI-Open/FlagAI/pull/303) 以及 AltDiffusion-m18模型 [#302](https://github.com/FlagAI-Open/FlagAI/pull/302); 
- [17 Mar 2023] 支持v1.6.2版本, 可以使用新的优化器 [#266](https://github.com/FlagAI-Open/FlagAI/pull/266), 并增加了英文gpt模型GPT2-base-en; 
- [2 Mar 2023] 支持v1.6.1版本, 增加Galactica模型 [#234](https://github.com/FlagAI-Open/FlagAI/pull/234), 大模型推理的低资源工具包BMInf [#238](https://github.com/FlagAI-Open/FlagAI/pull/238), 以及P-tuning样例 [#227](https://github.com/FlagAI-Open/FlagAI/pull/238)
- [12 Jan 2023] 发布v1.6.0版本, 新增支持并行训练库 [**BMTrain**](https://github.com/OpenBMB/BMTrain) 以及集成 [**Flash Attention**](https://github.com/HazyResearch/flash-attention) 到 Bert 和 Vit 模型提速端到端训练, 示例见 [FlashAttentionBERT](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/bert_title_generation_english/train_flash_atten.py)和 [FlashAttentionViT](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/vit_cifar100/train_single_gpu_flash_atten.py). 同时增加了基于对比搜索的文本生成方法 [**SimCTG**](https://github.com/yxuansu/SimCTG) 以及基于 AltDiffusion 进行 DreamBooth 个性化微调, 示例见 [AltDiffusionNaruto](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/AltDiffusion/dreambooth.py). 
- [28 Nov 2022] 发布v1.5.0版本, 支持1.1B参数的 [**EVA-CLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP) 以及[ALM: 基于GLM的阿拉伯语大模型], 示例见[**ALM**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/ALM)
- [10 Nov 2022] 发布v1.4.0版本, 支持[AltCLIP: 更改CLIP中的语言编码器以扩展语言功能](https://arxiv.org/abs/2211.06679v1), 示例见[**AltCLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP)以及[**AltDiffusion**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion)
- [29 Aug 2022] 支持v1.3.0版本, 增加CLIP模块以及重新设计了tokenizer的API: [#81](https://github.com/FlagAI-Open/FlagAI/pull/81)
- [21 Jul 2022] 支持v1.2.0版本, 支持ViT系列模型: [#71](https://github.com/FlagAI-Open/FlagAI/pull/71)
- [29 Jun 2022] 支持v1.1.0版本, 支持OPT的加载，微调和推理[#63](https://github.com/FlagAI-Open/FlagAI/pull/63)
- [17 May 2022] 做出了我们的第一份贡献[#1](https://github.com/FlagAI-Open/FlagAI/pull/1)

## 许可 LICENSE 


FlagAI飞智大部分项目基于 [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)，但是请注意部分项目代码基于其他协议：

* Megatron-LM 是基于协议 [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)
* GLM 是基于协议 [MIT license](https://github.com/THUDM/GLM/blob/main/LICENSE)
* AltDiffusion 是基于协议 [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

## 平台支持

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
