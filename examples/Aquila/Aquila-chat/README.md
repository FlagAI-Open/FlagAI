
# AquilaChat

## 简介/Overview
Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，在Aquila的训练过程中实现了比Magtron+DeepSpeed zero-2将近８倍的训练效率。Aquila语言大模型是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。

The Aquila language model inherits the architectural design advantages of GPT-3 and LLaMA, replacing a batch of more efficient underlying operator implementations and redesigning the tokenizer for Chinese-English bilingual support. It upgrades the BMTrain parallel training method, achieving nearly 8 times the training efficiency of Magtron+DeepSpeed ZeRO-2 in the training process of Aquila. The Aquila language model is trained from scratch on high-quality Chinese and English corpora. Through data quality control and various training optimization methods, it achieves better performance than other open-source models with smaller datasets and shorter training times. It is also the first large-scale open-source language model that supports Chinese-English-Knowledge, commercial licensing, and complies with domestic data regulations.
  
AquilaChat-7B是在Aquila-7B模型的基础上，进行SFT微调后的支持中英双语的对话式语言模型。AquilaChat-7B模型由智源研究院研发

AquilaChat-7B is a conversational language model that supports Chinese-English dialogue. It is based on the Aquila-7B model and fine-tuned using SFT. AquilaChat-7B model was developed by Beijing Academy of Artificial Intelligence. 


我们的模型也同时支持[Huggingface平台](https://huggingface.co/BAAI)。

We also support [Huggingface](https://huggingface.co/BAAI).

AquilaChat模型主要为了验证基础模型能力，您可以根据自己需要对模型进行使用，修改和商业化，但必须遵守所有国家的法律法规，并且对任何第三方使用者提供Aquila系列模型的来源以及Aquila系列模型协议的副本。

The AquilaChat model was primarily developed to verify the capabilities of the foundational model. You can use, modify, and commercialize the model according to your needs, but you must comply with all applicable laws and regulations in your country. Additionally, you must provide the source of the Aquila series models and a copy of the Aquila series model lincense to any third-party users.

## 模型细节/Model details

|   模型/Model          |  状态/State    | 能否商用/Commercial use?  |  所用显卡/GPU   |                                    
| :---------------- | :------- | :-- |:-- |   
| Aquila-7B         | 已发布  |   ✅   | Nvidia-A100  |  
| AquilaChat-7B          |已发布  |    ✅    | Nvidia-A100  | 
| AquilaCode-7B-NV          |已发布  |    ✅   |   Nvidia-A100   | 
| AquilaCode-7B-TS           |已发布 |   ✅    |  Tianshu-BI-V100   |
| Aquila-33B          | **敬请期待**  |   ✅   | Nvidia-A100  |
| AquilaChat-33B           |**敬请期待**  |    ✅    | Nvidia-A100  | 


我们使用了一系列更高效的底层算子来辅助模型训练，其中包括参考[flash-attention](https://github.com/HazyResearch/flash-attention)的方法并替换了一些中间计算，同时还使用了RMSNorm。在此基础上，我们应用了[BMtrain](https://github.com/OpenBMB/BMTrain)技术进行轻量化的并行训练，该技术采用了数据并行、ZeRO（零冗余优化器）、优化器卸载、检查点和操作融合、通信-计算重叠等方法来优化模型训练过程。

Aquila模型所采用的tokenizer是由我们从头开始训练的，支持中英双语。我们在处理英文、中文以及代码数据时，采用了不同的分词器对一万个样本进行了抽取。随后，我们统计了每个样本的token数量，并将其记录在表格中。Aquila tokenizer与其他tokenizer的参数对比见下表:

We used a series of more efficient low-level operators to assist with model training, including methods referenced from [flash-attention](https://github.com/HazyResearch/flash-attention) and replacing some intermediate calculations, as well as using RMSNorm. Building upon this foundation, we applied the [BMtrain](https://github.com/OpenBMB/BMTrain) for lightweight parallel training, which utilizes methods such as data parallelism, ZeRO (zero redundancy optimizer), optimizer offloading, checkpoint and operation fusion, and communication-computation overlap to optimize the model training process.

The tokenizer used in the Aquila model was trained from scratch by us and supports both English and Chinese. We used different tokenizers to extract ten thousand data samples from English, Chinese, and code data respectively, obtained the count of tokens for each sample, and also included it in the table. The parameters of this tokenizer are compared to those of other tokenizers in the table below:

| 模型/Model | 词表大小/Vocab size | 说明/Note |英文平均tokens量/Avg tokens(English)| 中文平均tokens量/Avg tokens(Chinesse)|代码平均tokens量/Avg tokens(code)  |
|  -----  | ----  | -----  | ----  | -----  | ----  | 
| GPT2 | 50527 | bpe|1717 | 1764|2323 |
| LlaMA | 32000 | sp(bpe)|1805| 1257|1970 |
| Aquila | 100000 | bpe|1575 | 477|1679 |


## 训练数据集/Training data 

我们采用了一系列高质量中英文数据集来训练和微调我们的对话语言模型，并且在不断更新迭代。

We used a series of high-quality Chinese and English datasets to train and fine-tune our conversational language model, and continuously updated it through iterations.

我们额外支持了两种多模态的指令: 文图生成和图片编辑，数据集格式请参考[这里](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/Aquila-chat/data/sft_samples.jsonl)

We have added support for two additional multimodal instructions: text-to-image generation and image editing. Please refer to the dataset format [here](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/Aquila-chat/data/sft_samples.jsonl).


## 使用方式/How to use

### 1. 推理/Inference

```python
import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.aquila import aquila_generate
from flagai.data.tokenizer import Tokenizer
import bminf

state_dict = "./checkpoints_in"
model_name = 'aquilachat-33b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
cache_dir = os.path.join(state_dict, model_name)
model.eval()
model.half()
model.cuda()

predictor = Predictor(model, tokenizer)

text = "北京为什么是中国的首都？"

def pack_obj(text):
    obj = dict()
    obj['id'] = 'demo'

    obj['conversations'] = []
    human = dict()
    human['from'] = 'human'
    human['value'] = text
    obj['conversations'].append(human)
    # dummy bot
    bot = dict()
    bot['from'] = 'gpt'
    bot['value'] = ''
    obj['conversations'].append(bot)

    obj['instruction'] = ''

    return obj

def delete_last_bot_end_singal(convo_obj):
    conversations = convo_obj['conversations']
    assert len(conversations) > 0 and len(conversations) % 2 == 0
    assert conversations[0]['from'] == 'human'

    last_bot = conversations[len(conversations)-1]
    assert last_bot['from'] == 'gpt'

    ## from _add_speaker_and_signal
    END_SIGNAL = "\n"
    len_end_singal = len(END_SIGNAL)
    len_last_bot_value = len(last_bot['value'])
    last_bot['value'] = last_bot['value'][:len_last_bot_value-len_end_singal]
    return

def convo_tokenize(convo_obj, tokenizer):
    chat_desc = convo_obj['chat_desc']
    instruction = convo_obj['instruction']
    conversations = convo_obj['conversations']
            
    # chat_desc
    example = tokenizer.encode_plus(f"{chat_desc}", None, max_length=None)['input_ids']
    EOS_TOKEN = example[-1]
    example = example[:-1] # remove eos
    # instruction
    instruction = tokenizer.encode_plus(f"{instruction}", None, max_length=None)['input_ids']
    instruction = instruction[1:-1] # remove bos & eos
    example += instruction

    for conversation in conversations:
        role = conversation['from']
        content = conversation['value']
        print(f"role {role}, raw content {content}")
        content = tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
        content = content[1:-1] # remove bos & eos
        print(f"role {role}, content {content}")
        example += content
    return example

print('-'*80)
print(f"text is {text}")

from cyg_conversation import default_conversation

conv = default_conversation.copy()
conv.append_message(conv.roles[0], text)
conv.append_message(conv.roles[1], None)

tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
tokens = tokens[1:-1]

with torch.no_grad():
    out = aquila_generate(tokenizer, model, [text], max_gen_len:=200, top_p=0.95, prompts_tokens=[tokens])
    print(f"pred is {out}")


```

### 2. 可监督微调/Supervised Fine-tuning(SFT)
#### Step 1: 配置模型/ Setup Checkpoints
在`./checkpoints_in`里新建`aquilachat-7b`目录。将微调后的checkpoint，以及原始`aquilachat-7b`模型里的其余文件，包括`config.json`, `mergex.txt`, `vocab.json`, `special_tokens_map.json`放进去

Create a new directory named `aquilachat-7b` inside `./checkpoints_in`. Place the fine-tuned checkpoint and all other files from the original `aquilachat-7b` model, including `config.json`, `mergex.txt`, `vocab.json`, and `special_tokens_map.json`, into this directory.

#### Step 2: 修改参数/ Modify Parameters
* `cd /examples/Aquila/Aquila-sft`
* 配置`hostfile`文件, 参考[这里](../../../doc_zh/TUTORIAL_8_ENVIRONMENT_SETUP.md#a配置hostfilehostfile-中的v100-1-与sshconfig-对应) ; Configure the `hostfile` file, refer to [here](../../../docs/TUTORIAL_8_ENVIRONMENT_SETUP.md)
* 配置`bmtrain_mgpu.sh`文件, 将`SCRIPT_FILE`改成`aquila_sft.py`; configure the `bmtrain_mgpu.sh` file, change `SCRIPT_FILE` to `aquila_sft.py`
* (可选) 在`Aquila-sft.yaml`文件里更改参数 ; (optional) change parameters in `Aquila-sft.yaml`

| 参数名 Parameter             | 类型 Type | 描述 Description                                        |
|--------------------------------|------------|-------------------------------------------------------|
| batch_size | int   | 每次迭代训练时，从数据集中抽取的样本数。一般来说，它越大，处理速度越快，但会占用更多的内存; The number of samples extracted from the dataset for each iteration during training. Generally, a larger batch size can speed up processing but may also consume more memory                    |
| gradient_accumulation_steps | int   | 在更新模型权重之前，要对多个小批次进行梯度计算的次数。主要应用于GPU显存较小的情况下，可以使用小的batch_size，通过梯度累积达到与大batch_size相同的效果; The number of samples extracted from the dataset for each iteration during training. Generally, a larger batch size can speed up processing but may also consume more memoryimages                  |
| lr | float   | 指控制模型更新参数时的步长或速率。学习率过高可能导致模型不收敛，而学习率过低则可能导致训练时间过长或者陷入局部最优解; The step size or rate at which the model updates its parameters during training. A high learning rate may cause the model not to converge, while a low learning rate may result in long training times or being stuck in a local optimum                  |
| warm_up | float   | 初始学习率与原始学习率的比例; The ratio between the initial learning rate and the original learning rate
| save_interval | int  | 模型保存的间隔，即每训练多少个iteration保存一次模型。当训练时间较长时，保存间隔可以避免因突然中断或出现错误导致训练成果全部丢失; The interval at which the model is saved, i.e., how often the model is saved per epoch during training. When training takes a long time, saving intervals can prevent all training achievements from being lost due to sudden interruptions or errors.                    |
| enable_sft_conversations_dataset_v3 | bool  | 数据处理方式; Data preprocessing method                    |
| enable_sft_dataset_dir | str   | 可监督微调的数据集目录; Dataset directory of SFT dataset                    |
| enable_sft_dataset_file | str   | 可监督微调的数据集文件名; Filename of SFT dataset                     |                  |




#### Step 3: 启动可监督微调/Start SFT
```
bash dist_trigger_docker.sh hostfile Aquila-sft.yaml aquilachat-7b [实验名]
```
接下来会输出下列信息，注意`NODES_NUM`应该与节点数相等，`LOGFILE`是模型运行的日志文件；The following information will be output. Note that `NODES_NUM` should be equal to the number of nodes, and `LOGFILE` is the log file for the model run.

![Screenshot](../img/info.jpg)

成功训练之前能看到如下信息(具体参数可能不同)； Before successful training, you may see the following information with parameters that may differ:

![Screenshot](../img/info2.jpg)


## 证书/License

`AquilaChat`系列开源模型使用 [智源Aquila系列模型许可协议](../../../BAAI_Aquila_Model_License.pdf), 原始代码基于[Apache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0)。


`AquilaChat` open-source model is licensed under [ BAAI Aquila Model Licence Agreement](../../../BAAI_Aquila_Model_License.pdf). The source code is under [Apache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0).
