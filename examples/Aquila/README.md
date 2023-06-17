
![Aquila_logo](./img/Aquila.PNG)



# 悟道·天鹰（Aquila）

悟道·天鹰（Aquila） 语言大模型是首个具备中英双语知识、支持商用许可协议、国内数据合规需求的开源语言大模型。
- 🌟 支持开源商用许可。Aquila系列模型的源代码基于 Apache 2.0 协议，模型权重基于《智源Aquila系列模型许可协议》，使用者在满足许可限制的情况下，可用于商业目的。
- ✍️ 具备中英文知识。Aquila系列模型在中英文高质量语料基础上从 0 开始训练，中文语料约占 40%，保证模型在预训练阶段就开始积累原生的中文世界知识，而非翻译而来的知识。
- 👮‍♀️符合国内数据合规需求。Aquila系列模型的中文语料来自智源多年积累的中文数据集，包括来自1万多个站源的中文互联网数据（其中99%以上为国内站源），以及获得国内权威机构支持的高质量中文文献数据、中文书籍数据等。我们仍在持续积累高质量、多样化的数据集，并源源不断加入Aquila基础模型后续训练中。
- 🎯持续迭代，持续开源开放。我们将不断完善训练数据、优化训练方法、提升模型性能，在更优秀的基础模型基座上，培育枝繁叶茂的“模型树”，持续开源开放更新的版本。


悟道 · 天鹰 Aquila 模型的更多细节将在官方技术报告中呈现，预计 2023 年 6 月底发布。请关注官方渠道更新。包括 [FlagAI GitHub仓库](https://github.com/FlagAI-Open/FlagAI/)，[FlagAI 知乎账号](https://www.zhihu.com/people/95-22-20-18)、[FlagAI 官方技术交流群](https://github.com/FlagAI-Open/FlagAI/blob/master/wechat-qrcode.jpg)、智源研究院微信公众号、智源社区微信公众号。


|   模型          |  模型类型    | 简介  |  文件路径   |   单独下载模型权重  |  状态   |  训练所用显卡   |                                   
| :---------------- | :------- | :-- |:-- |   :-- | :-- | :-- | 
| Aquila-7B         | 基础模型，70亿参数  |   Aquila 基础模型在技术上继承了 GPT-3、LLaMA 等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的 tokenizer，升级了 BMTrain 并行训练方法，实现了比 Magtron+DeepSpeed ZeRO-2 将近８倍的训练效率。   | [./examples/Aquila/Aquila-pretrain](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-pretrain)  | [model.baai.ac.cn/model-detail/100098](http://model.baai.ac.cn/model-detail/100098) | 已发布 | Nvidia-A100 |
| Aquila-33B          |已发布  |    同上    | Nvidia-A100  | 
| AquilaChat-7B          |已发布  |    ✅   |   Nvidia-A100   | 
| AquilaChat-33B           |已发布 |   ✅    |  Tianshu-BI-V100   |
| AquilaCode-7B-NV          | **敬请期待**  |   ✅   | Nvidia-A100  |
| AquilaCode-7B-TS           |**敬请期待**  |    ✅    | Nvidia-A100  | 


<table>
  <tr>
    <td>模型</td>
    <td>模型类型</td>
    <td>简介</td>
    <td style="width: 100px;">文件路径</td>
    <td style="width: 100px;">单独下载模型权重</td>
    <td>状态</td>
    <td>训练所用显卡</td>
  </tr>
  <tr>
    <td>Aquila-7B</td>
    <td>基础模型，70亿参数</td>
    <td rowspan="2">Aquila 基础模型在技术上继承了 GPT-3、LLaMA 等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的 tokenizer，升级了 BMTrain 并行训练方法，实现了比 Magtron+DeepSpeed ZeRO-2 将近８倍的训练效率。</td>
    <td>https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-pretrain</td>
    <td>http://model.baai.ac.cn/model-detail/100098</td>
    <td>已发布</td>
    <td>Nvidia-A100</td>
  </tr>
  <tr>
    <td>14</td>
    <td>15</td>
    <td>16</td>
    <td>17</td>
    <td>18</td>
    <td>19</td>
  </tr>
  <tr>
    <td>20</td>
    <td>21</td>
    <td rowspan="2">22</td>
    <td>24</td>
    <td>25</td>
    <td>26</td>
    <td>27</td>
  </tr>
  <tr>
    <td>28</td>
    <td>29</td>
    <td>30</td>
    <td>31</td>
    <td>32</td>
    <td>33</td>
  </tr>
  <tr>
    <td>34</td>
    <td>35</td>
    <td rowspan="2">36</td>
    <td>37</td>
    <td>38</td>
    <td>39</td>
    <td>39</td>
  </tr>
  <tr>
    <td>40</td>
    <td>41</td>
    <td rowspan="3">42</td>
    <td>43</td>
    <td>44</td>
    <td>46</td>
  </tr>


</table>





## 简介/Overview
Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，在Aquila的训练过程中实现了比Magtron+DeepSpeed zero-2将近８倍的训练效率。Aquila语言大模型是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。

The Aquila language model inherits the architectural design advantages of GPT-3 and LLaMA, replacing a batch of more efficient underlying operator implementations and redesigning the tokenizer for Chinese-English bilingual support. It upgrades the BMTrain parallel training method, achieving nearly 8 times the training efficiency of Magtron+DeepSpeed ZeRO-2 in the training process of Aquila. The Aquila language model is trained from scratch on high-quality Chinese and English corpora. Through data quality control and various training optimization methods, it achieves better performance than other open-source models with smaller datasets and shorter training times. It is also the first large-scale open-source language model that supports Chinese-English-Knowledge, commercial licensing, and complies with domestic data regulations.


<!-- 我们同时也支持[Huggingface平台](https://huggingface.co/BAAI)。

We also support [Huggingface](https://huggingface.co/BAAI). -->

最低硬件需求：运行Aquila-7B系列需要内存30G, 显存18G，生成最大长度 2048 tokens。

Minimum hardware requirements for running the Aquila-7b series, you need at least 30GB of memory and 18GB of GPU memory, and the maximum length of text generated should be 2048 tokens.

## 模型细节/Model details

|   模型/Model          |  状态/State    | 能否商用/Commercial use?  |  所用显卡/GPU   |                                    
| :---------------- | :------- | :-- |:-- |   
| Aquila-7B         | 已发布  |   ✅   | Nvidia-A100  |  
| AquilaChat-7B          |已发布  |    ✅    | Nvidia-A100  | 
| AquilaCode-7B-NV          |已发布  |    ✅   |   Nvidia-A100   | 
| AquilaCode-7B-TS           |已发布 |   ✅    |  Tianshu-BI-V100   |
| Aquila-33B          | **敬请期待**  |   ✅   | Nvidia-A100  |
| AquilaChat-33B           |**敬请期待**  |    ✅    | Nvidia-A100  | 

我们使用了一系列更高效的底层算子来辅助模型训练，其中包括参考[flash-attention](https://github.com/HazyResearch/flash-attention)的方法并替换了一些中间计算，同时还使用了RMSNorm。在此基础上，我们升级了[BMtrain](https://github.com/OpenBMB/BMTrain)技术进行轻量化的并行训练，该技术采用了数据并行、ZeRO（零冗余优化器）、优化器卸载、检查点和操作融合、通信-计算重叠等方法来优化模型训练过程。

Aquila模型所采用的tokenizer是由我们从头开始训练的，支持中英双语。我们在处理英文、中文以及代码数据时，采用了不同的分词器对一万个样本进行了抽取。随后，我们统计了每个样本的token数量，并将其记录在表格中。Aquila tokenizer与其他tokenizer的参数对比见下表:

We used a series of more efficient low-level operators to assist with model training, including methods referenced from [flash-attention](https://github.com/HazyResearch/flash-attention) and replacing some intermediate calculations, as well as using RMSNorm. Building upon this foundation, we applied the [BMtrain](https://github.com/OpenBMB/BMTrain) for lightweight parallel training, which utilizes methods such as data parallelism, ZeRO (zero redundancy optimizer), optimizer offloading, checkpoint and operation fusion, and communication-computation overlap to optimize the model training process.

The tokenizer used in the Aquila model was trained from scratch by us and supports both English and Chinese. We used different tokenizers to extract ten thousand data samples from English, Chinese, and code data respectively, obtained the count of tokens for each sample, and also included it in the table. The parameters of this tokenizer are compared to those of other tokenizers in the table below:



| 模型/Model | 词表大小/Vocab size | 说明/Note |英文平均tokens量/Avg tokens(English)| 中文平均tokens量/Avg tokens(Chinesse)|代码平均tokens量/Avg tokens(code)  |
|  -----  | ----  | -----  | ----  | -----  | ----  | 
| GPT2 | 50527 | bpe|1717 | 1764|2323 |
| LlaMA | 32000 | sp(bpe)|1805| 1257|1970 |
| Aquila | 100000 | bpe|1575 | 477|1679 |


## 训练数据集/Training data 
Aquila预训练使用了Pile，[RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T), [Wikipedia](https://huggingface.co/datasets/wikipedia), [C4](https://huggingface.co/datasets/c4), 悟道中文数据集、电子书、专利、百科、论坛, github数据等, 详情可见下图。

The Aquila-7B model was pretrained on Pile，[RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T), [Wikipedia](https://huggingface.co/datasets/wikipedia), [C4](https://huggingface.co/datasets/c4), Wudao Corpus、e-book、Patent, encyclopedia, forum, github etc. Details are given in the figure below.

<!-- ![Screenshot](./img/data_dist.png) -->
Aquila 系列模型的预训练数据不开源，但数据分布情况将在官方技术报告中展现（预计6月底发布，敬请期待）。

The pre-training data of the Aquila series models are not open-sourced, but the data distribution will be presented in the official technical report (expected to be released by the end of June, stay tuned).


## 使用方式/How to use

### 1. 预训练/Pre-training
#### Step 1: 修改参数/Modify Parameters

* `cd /examples/aquila`
* 配置`hostfile`文件, 参考[这里](../../doc_zh/TUTORIAL_8_ENVIRONMENT_SETUP.md#a配置hostfilehostfile-中的v100-1-与sshconfig-对应) ; Configure the `hostfile` file, refer to [here](../../docs/TUTORIAL_8_ENVIRONMENT_SETUP.md)
* 配置`bmtrain_mgpu.sh`文件, 将`SCRIPT_FILE`改成`aquila_pretrain.py`; configure the `bmtrain_mgpu.sh` file, change `SCRIPT_FILE` to `aquila_pretrain.py`
* (可选) 在`Aquila-pretrain.yaml`文件里更改参数 ; (optional) change parameters in `Aquila-pretrain.yaml`

| 参数名 Parameter             | 类型 Type | 描述 Description                                        |
|--------------------------------|------------|-------------------------------------------------------|
| batch_size | int   | 每次迭代训练时，从数据集中抽取的样本数。一般来说，它越大，处理速度越快，但会占用更多的内存; The number of samples extracted from the dataset for each iteration during training. Generally, a larger batch size can speed up processing but may also consume more memory                    |
| gradient_accumulation_steps | int   | 在更新模型权重之前，要对多个小批次进行梯度计算的次数。主要应用于GPU显存较小的情况下，可以使用小的batch_size，通过梯度累积达到与大batch_size相同的效果; The number of samples extracted from the dataset for each iteration during training. Generally, a larger batch size can speed up processing but may also consume more memoryimages                  |
| lr | float   | 指控制模型更新参数时的步长或速率。学习率过高可能导致模型不收敛，而学习率过低则可能导致训练时间过长或者陷入局部最优解; The step size or rate at which the model updates its parameters during training. A high learning rate may cause the model not to converge, while a low learning rate may result in long training times or being stuck in a local optimum                  |
| warm_up | float   | 初始学习率与原始学习率的比例; The ratio between the initial learning rate and the original learning rate
| save_interval | int  | 模型保存的间隔，即每训练多少个iteration保存一次模型。当训练时间较长时，保存间隔可以避免因突然中断或出现错误导致训练成果全部丢失; The interval at which the model is saved, i.e., how many iterations the model is saved during training. When training takes a long time, saving intervals can prevent all training achievements from being lost due to sudden interruptions or errors.                    |

* 我们的演示数据集放在`../indexed_dataset/data/demo_text_document`里。 如果想修改预训练数据集，可更改`aquila_pretrain.py`里的`data_prefix`参数; Our demo dataset is located in `../indexed_dataset/data/demo_text_document`. If you want to modify the pre-training dataset, you can change the data_prefix parameter in `aquila_pretrain.py`.
#### Step 2: 启动训练/Start training
对于Aquila-7B模型
```
bash dist_trigger_docker.sh hostfile Aquila-pretrain.yaml aquila-7b [实验名]
```   

对于Aquila-33B模型
```
bash dist_trigger_docker.sh hostfile Aquila-pretrain-33B.yaml aquila-33b [实验名]
```   

 接下来会输出下列信息，注意`NODES_NUM`应该与节点数相等，`LOGFILE`是模型运行的日志文件；The following information will be output. Note that `NODES_NUM` should be equal to the number of nodes, and `LOGFILE` is the log file for the model run.

![Screenshot](./img/info.jpg)

成功训练之前能看到如下信息(具体参数可能不同)； Before successful training, you may see the following information with parameters that may differ:

![Screenshot](./img/info2.jpg)
  
### 2. 可监督微调/Supervised Fine-tuning(SFT)
#### Step 1: 修改参数/Modify Parameters
* `cd /examples/aquila`
* 配置`hostfile`文件, 参考[这里](../../doc_zh/TUTORIAL_8_ENVIRONMENT_SETUP.md#a配置hostfilehostfile-中的v100-1-与sshconfig-对应) ; Configure the `hostfile` file, refer to [here](../../docs/TUTORIAL_8_ENVIRONMENT_SETUP.md)
* 配置`bmtrain_mgpu.sh`文件, 将`SCRIPT_FILE`改成`aquila_pretrain.py`; configure the `bmtrain_mgpu.sh` file, change `SCRIPT_FILE` to `aquila_pretrain.py`
* (可选) 在`Aquila-pretrain.yaml`文件里更改参数 ; (optional) change parameters in `Aquila-pretrain.yaml`



#### Step 2: 启动可监督微调/Start SFT
```
cd ../Aquila-chat/
```
对于Aquila-7B模型：
```
bash dist_trigger_docker.sh hostfile Aquila-chat.yaml aquila-7b [实验名 experiment name]
```
对于Aquila-33B模型:
```
bash dist_trigger_docker.sh hostfile Aquila-chat.yaml aquila-33b [实验名 experiment name]
```
接下来会输出下列信息，注意`NODES_NUM`应该与节点数相等，`LOGFILE`是模型运行的日志文件；The following information will be output. Note that `NODES_NUM` should be equal to the number of nodes, and `LOGFILE` is the log file for the model run.

![Screenshot](./img/info.jpg)

成功训练之前能在日志里看到如下信息(具体参数可能不同)； Before successful training, you may see the following information in the log file with parameters that may differ:

![Screenshot](./img/info2.jpg)

### 3. 推理/Inference

```python
import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.data.tokenizer import Tokenizer
import bminf

state_dict = "./checkpoints_in/"
model_name = 'aquila-7b' # 'aquila-33b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()

model.eval()
model.half()
model.cuda()

predictor = Predictor(model, tokenizer)

text = "北京在哪儿?"
text = f'{text}' 
print(f"text is {text}")
with torch.no_grad():
    out = predictor.predict_generate_randomsample(text, out_max_length=200, temperature=0)
    print(f"pred is {out}")

```




## 证书/License

Aquila-7B和Aquila-33B开源模型使用 [智源Aquila系列模型许可协议](../../BAAI_Aquila_Model_License.pdf), 原始代码基于[Apache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0)。


Aquila-7B and Aquila-33B open-source model is licensed under [ BAAI Aquila Model Licence Agreement](../../BAAI_Aquila_Model_License.pdf). The source code is under [Apache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0)
