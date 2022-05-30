![FlagAI](logo.png)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6052/badge)](https://bestpractices.coreinfrastructure.org/projects/6052)
[![Python package](https://github.com/BAAI-Open/FlagAI/actions/workflows/python-package.yml/badge.svg)](https://github.com/BAAI-Open/FlagAI/actions/workflows/python-package.yml)
[English](README.md)

--------------------------------------------------------------------------------

飞智是一个快速、易于使用和可扩展的大模型工具包。 我们的目标是支持在多模态的各种下游任务上训练、微调和部署大规模模型。 目前，我们专注于 NLP 模型和任务。 在不久的将来，我们将支持其他模态。
<br><br>

* 现在它支持最高百亿参数的**悟道GLM**(详见[GLM介绍](/doc_zh/GLM.md))。它同时也支持**BERT**、**RoBERTa**、**GPT2**、**T5** 模型和 Huggingface Transformers 的模型。

* 它提供 API 以快速下载并在给定（中/英文）文本上使用这些预训练模型，在您自己的数据集上对其进行微调(fine-tuning)或者应用[提示学习(prompt-tuning)](/doc_zh/TUTORIAL_7_PROMPT_LERANING.md)，然后在我们的模型中心与社区共享它们。 

* 这些模型可以应用于文本，用于文本分类、信息提取、问答、摘要、文本生成等任务，尤其是中文。

* 飞智由三个最流行的数据/模型并行库（[PyTorch](https://pytorch.org/)/[Deepspeed](https://www.deepspeed.ai/)/[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)）提供支持，它们之间实现了无缝集成。 你可以用不到十行代码来并行你的训练/测试过程。


本项目的部分代码基于[GLM](https://github.com/THUDM/GLM),[Transformers](https://github.com/huggingface/transformers) 和 [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples).

<!-- toc -->

- [安装](#安装)
- [快速上手](#快速上手)
    - [加载模型和分词器](#加载模型和分词器)
    - [使用预测器](#使用预测器)
    - [命名实体识别任务示例](#命名实体识别任务示例 )
    - [标题生成任务示例](#标题生成任务示例)
    - [语义相似度匹配任务示例](#语义相似度匹配任务示例)
- [预训练模型以及样例](#预训练模型以及样例)
- [教程](#教程)
- [了解更多关于FlagAI](#了解更多关于FlagAI)
- [贡献代码](#贡献代码)
- [许可证](#许可证)

<!-- tocstop -->
# 安装
* PyTorch version >= 1.8.0
* Python version >= 3.8
* 使用GPUs进行训练和测试, 你需要安装CUDA 和 NCCL

通过`pip`安装:
```shell
pip install -U flagai
```

- [可选]下载源码安装:

```shell
git clone https://github.com/BAAI-Open/FlagAI.git
python setup.py install
```

- [可选] 开启训练加速需要安装 NVIDIA's [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- [可选] 使用 ZeRO 优化器，需要安装 [DEEPSPEED](https://github.com/microsoft/DeepSpeed)
```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .
ds_report # 检查deepspeed的状态
```
- [提示] 单节点docker环境下, 运行多卡数据并行需要设置host. 例如，docker节点 root@127.0.0.1，其端口 7110。
```
>>> vim ~/.ssh/config
Host 127.0.0.1
    Hostname 127.0.0.1
    Port 7110
    User root
```
- [提示] 多节点环境, 需要生成 ssh keys 并拷贝公钥到所有节点 (in `~/.ssh/`)
```
>>> ssh-keygen -t rsa -C "xxx@xxx.com"
```


# 快速上手
我们提供了精选的中英文预训练模型，以及经过训练可以执行不同任务的模型权重。 您可以通过 AutoLoader 加载这些模型以进行训练和预测。更多样例见 `FlagAI/quickstart`。

## 加载模型和分词器
我们提供 `AutoLoad` 类来快速加载模型和分词器，例如：
```python
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(task_name="title-generation",
                         model_name="RoBERTa-base-ch",
                         load_pretrain_params=True,
                         class_num=2)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
这个例子是针对`classification`任务的(分类），你也可以通过修改`task_name`来为其他任务建模。
`class_num` 是分类任务的类别数。 然后您可以使用模型和标记器进行微调或测试。

## 使用预测器
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


## 命名实体识别任务示例

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


## 语义相似度匹配任务示例

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

# 预训练模型以及样例
* [GLM-large-ch用户完形填空问答](/doc_zh/TUTORIAL_11_GLM_BLANK_FILLING_QA.md)
* [GLM-large-ch用于诗歌生成](doc_zh/TUTORIAL_13_GLM_EXAMPLE_PEOTRY_GENERATION.md)
* [GLM-large-ch用于标题生成](doc_zh/TUTORIAL_12_GLM_EXAMPLE_TITLE_GENERATION.md)
* [对 huggingface t5-11b 模型的支持 以及加速的tricks](doc_zh/TUTORIAL_14_HUGGINGFACE_T5.md)
* [RoBERTa-base-ch用于标题生成](doc_zh/TUTORIAL_15_BERT_EXAMPLE_TITLE_GENERATION.md)
* [RoBERTa-base-ch用于语义相似度匹配](doc_zh/TUTORIAL_16_BERT_EXAMPLE_SEMANTIC_MATCHING.md)
* [RoBERTa-base-ch用于命名实体识别](/doc_zh/TUTORIAL_17_BERT_EXAMPLE_NER.md)
* [GPT-2用于文本续写](/doc_zh/TUTORIAL_18_GPT2_WRITING.md)
* [T5用于标题生成](/doc_zh/TUTORIAL_19_T5_EXAMPLE_TITLE_GENERATION.md)
* [用GLM10b模型在TNEWS短文本分类数据集上微调](doc_zh/TUTORIAL_20_GLM_TNEWS.md)


本节解释了本项目中基础NLP类是如何工作的，如何加载预先训练的模型来标记您的文本，如何使用不同的词或文档嵌入来得到表示，以及如何训练自己的语言模型、序列标注模型和文本分类模型。更多样例见 `FlagAI/examples`。


# 教程
我们提供了一组教程来帮助您快速上手使用本库：
* [Tutorial 1: 如何构建和应用分词器](/doc_zh/TUTORIAL_1_TOKENIZER.md)
* [Tutorial 2: 数据集预处理流程](/doc_zh/TUTORIAL_2_DATASET.md)
* [Tutorial 3: 模型的主要功能及相关结构](/doc_zh/TUTORIAL_3_MODEL.md)
* [Tutorial 4: 为模型和数据并行训练定制训练器](/doc_zh/TUTORIAL_4_TRAINER.md)
* [Tutorial 5: 使用 Autoloader 简化模型和分词器初始化过程](/doc_zh/TUTORIAL_5_INSTRUCTIONS_FOR_AutoLoader.md)
* [Tutorial 6: 将现成的推理算法与 Predictor 结合使用](/doc_zh/TUTORIAL_6_INSTRUCTIONS_FOR_PREDICTOR.md)
* [Tutorial 7: 使用飞智提示学习工具包来提高在SuperGLUE任务上的表现](/doc_zh/TUTORIAL_7_PROMPT_LERANING.md)
* [Tutorial 8: 多机训练模型搭建环境](/doc_zh/TUTORIAL_8_ENVIRONMENT_SETUP.md)
* [Tutorial 9: 使用encoder/decoder/encoder-decoder模型进行文本生成](/doc_zh/TUTORIAL_9_SEQ2SEQ_METHOD.md)
* [Tutorial 10: 转化一个模型为Megatron-LM的模型并行版本](/doc_zh/TUTORIAL_10_MEGATRON.md)


# 贡献代码
感谢您对贡献的兴趣！ 参与的方式有很多； 从我们的[贡献者指南](CONTRIBUTING.md) 开始，然后检查这些[未解决的问题](https://github.com/BAAI-WuDao/Sailing/issues)以执行特定任务。

# 联系我们
欢迎扫码加入FlagAI用户群

<img src="./flagai_wechat.png" width = "200" height = "200"  align=center />


# [许可证](/LICENSE)
大部分的FlagAI项目是基于[Apache 2.0 license](LICENSE), 但是部分的代码是基于其他的协议:

* Megatron-LM 是基于协议[Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)
* GLM 是基于协议[MIT license](https://github.com/THUDM/GLM/blob/main/LICENSE)
