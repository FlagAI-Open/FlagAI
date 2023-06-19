
![Aquila_logo](../img/Aquila.PNG)



# 悟道·天鹰（Aquila）

悟道·天鹰（Aquila） 语言大模型是首个具备中英双语知识、支持商用许可协议、国内数据合规需求的开源语言大模型。
- 🌟 **支持开源商用许可**。Aquila系列模型的源代码基于 [Apache 2.0 协议](https://www.apache.org/licenses/LICENSE-2.0)，模型权重基于[《智源Aquila系列模型许可协议》](../../../BAAI_Aquila_Model_License.pdf)，使用者在满足许可限制的情况下，可用于商业目的。
- ✍️ **具备中英文知识**。Aquila系列模型在中英文高质量语料基础上从 0 开始训练，中文语料约占 40%，保证模型在预训练阶段就开始积累原生的中文世界知识，而非翻译而来的知识。
- 👮‍♀️**符合国内数据合规需求**。Aquila系列模型的中文语料来自智源多年积累的中文数据集，包括来自1万多个站源的中文互联网数据（其中99%以上为国内站源），以及获得国内权威机构支持的高质量中文文献数据、中文书籍数据等。我们仍在持续积累高质量、多样化的数据集，并源源不断加入Aquila基础模型后续训练中。
- 🎯**持续迭代，持续开源开放**。我们将不断完善训练数据、优化训练方法、提升模型性能，在更优秀的基础模型基座上，培育枝繁叶茂的“模型树”，持续开源开放更新的版本。

**Read this in [English](./README_en.md).**


悟道 · 天鹰 Aquila 模型的更多细节将在官方技术报告中呈现。请关注官方渠道更新。包括 [FlagAI GitHub仓库](https://github.com/FlagAI-Open/FlagAI/)，[FlagAI 知乎账号](https://www.zhihu.com/people/95-22-20-18)、[FlagAI 官方技术交流群](https://github.com/FlagAI-Open/FlagAI/blob/master/wechat-qrcode.jpg)、智源研究院微信公众号、智源社区微信公众号。


|   模型          |  模型类型    | 简介  |  文件路径   |   单独下载模型权重  |  状态   |  训练所用显卡   |                                   
| :---------------- | :------- | :-- |:-- |   :-- | :-- | :-- | 
| Aquila-7B         | 基础模型，70亿参数  |   **Aquila 基础模型**在技术上继承了 GPT-3、LLaMA 等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的 tokenizer，升级了 BMTrain 并行训练方法，实现了比 Magtron+DeepSpeed ZeRO-2 将近８倍的训练效率。   | [./examples/Aquila/Aquila-pretrain](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-pretrain)  | [下载Aquila-7B](http://model.baai.ac.cn/model-detail/100098) | 已发布 | Nvidia-A100 |
| Aquila-33B          |基础模型，70亿参数  |    同上    | ——  | ——  | **敬请期待** | Nvidia-A100 | 
| AquilaChat-7B          |SFT model，基于 Aquila-7B 进行微调和强化学习  |    **AquilaChat 对话模型**支持流畅的文本对话及多种语言类生成任务，通过定义可扩展的特殊指令规范，实现 AquilaChat对其它模型和工具的调用，且易于扩展。 <br><br>例如，调用智源开源的 **[AltDiffusion](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion-m18) 多语言文图生成模型**，实现了流畅的文图生成能力。配合智源 **InstructFace 多步可控文生图模型**，轻松实现对人脸图像的多步可控编辑。  |   [./examples/Aquila/Aquila-chat](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-chat)  | [下载AquilaChat-7B](https://model.baai.ac.cn/model-detail/100101) | 已发布  | Nvidia-A100  | 
| AquilaChat-33B           |SFT model，基于 Aquila-33B 进行微调和强化学习 |   同上    |   ——    |——  | **敬请期待** | Nvidia-A100 | 
| AquilaCode-7B-NV          | 基础模型，“文本-代码”生成模型，基于 Aquila-7B继续预训练，在英伟达芯片完成训练  |   AquilaCode-7B 以小数据集、小参数量，实现高性能，是目前支持中英双语的、性能最好的开源代码模型，经过了高质量过滤、使用有合规开源许可的训练代码数据进行训练。<br><br> AquilaCode-7B 分别在英伟达和国产芯片上完成了代码模型的训练。  | [./examples/Aquila/Aquila-code](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-code)  |[下载AquilaCode-7B-NV](https://model.baai.ac.cn/model-detail/100102)  | 已发布  | Nvidia-A100 | 
| AquilaCode-7B-TS           |基础模型，“文本-代码”生成模型，基于 Aquila-7B继续预训练，在天数智芯芯片上完成训练  |    同上    | [./examples/Aquila/Aquila-code](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-code)  | [下载AquilaCode-7B-TS](https://model.baai.ac.cn/model-detail/100099)  | 已发布  | Tianshu-BI-V100  | 

<br>如有使用问题请先查看 [FAQ](https://github.com/FlagAI-Open/FlagAI/issues/371)，若不能解决，请直接提交 [issue](https://github.com/FlagAI-Open/FlagAI/issues) ~


## 快速开始使用 AquilaChat-7B 对话模型

### 基础模型的环境准备

1. 在本地克隆FlagAI github仓库
   
    ```
    git clone https://github.com/FlagAI-Open/FlagAI.git
    ```

2. 进入仓库，从源码安装FlagAI
   
    ```
    cd FlagAI
    python setup.py install
    ```
    注：我们目前支持在Ubuntu, Mac和Mac上运行，详细环境依赖信息可参考 [FlagAI环境安装](../../../README.md#requirements-and-installation)

3. 进入**AquilaChat-7B对话模型**目录
    ```
    cd examples/Aquila/Aquila-chat
    ```
对于AquilaChat-7B模型，我们提供**模型推理**, **微调**两种使用方式：

### 对话模型推理

正常模型推理(显存资源消耗约为14.6GB)：
```
python generate_chat.py
```
使用[BMInf](https://github.com/OpenBMB/BMInf)进行低资源推理(可调整所用内存)
```
python generate_chat_bminf.py
```
默认参数下显存资源消耗为4.3GB，可通过memory_limit参数手动设置最大资源消耗，如下图所示(2 << 30 代表2GB)：
![bminf](../img/bminf.png)

推理程序运行完毕之后，Aquila-7B模型会自动下载到`./checkpoints_in`里

<details><summary>示例输出如下：</summary>

模型对于示例prompt"汽车EDR是什么"给出随机回复

![aquila_generate](../img/aquila_generate.png)

</details>
注意：Aquila-7B基础模型用来做对话推理效果不如可监督微调后的AquilaChat-7B对话模型。

### 基础模型微调-SFT

1. 配置`hostfile`文件
    <details><summary>详情如下：</summary>
    以单机八卡为例
    1. 查看本机ip地址
            ```
            ifconfig eth0 | grep "inet " | awk '{print $2}'
            ```
    2. 在`hostfile`里填入
            ```
            [上一步得到的ip地址] slots=8
            ```
    3. 确认本机可以免密登录,可用如下指令测试
            ```
            ssh localhost
            ```
    
    </details>

2. 启动训练脚本
    ```
    bash dist_trigger_docker.sh hostfile Aquila-chat.yaml aquilachat-7b aquila_experiment
    ```
    **如果想启动LoRA微调，上一步改为运行**
    ```
    bash dist_trigger_docker.sh hostfile Aquila-chat-lora.yaml aquilachat-7b aquila_experiment
    ```

<details><summary>正确运行输出信息如下所示：</summary>

首先会输出下列信息，注意`NODES_NUM`应该与节点数相等，`LOGFILE`是模型运行的日志文件。

![Screenshot](../img/info.jpg)

成功训练之前能在日志里看到如下信息(具体参数可能不同)。

![Screenshot](../img/info2.jpg)

</details>


### 调整参数

对于以上示例，您可以通过修改下列参数来达到不同的训练和推理效果：

🌟执行预训练和微调任务前，可在训练脚本中的yaml文件里修改参数

|   参数名          |  类型   | 描述  |                                  
| :---------------- | :------- | :-- |   
| batch_size         | int  |   每次迭代训练时，从数据集中抽取的样本数。一般来说，它越大，处理速度越快，但会占用更多的内存;   |
| gradient_accumulation_steps          |int  |    在更新模型权重之前，要对多个小批次进行梯度计算的次数。主要应用于GPU显存较小的情况下，可以使用小的batch_size，通过梯度累积达到与大batch_size相同的效果;     |
| lr          |float  |    指控制模型更新参数时的步长或速率。学习率过高可能导致模型不收敛，而学习率过低则可能导致训练时间过长或者陷入局部最优解;    |   
| warm_up           |float |   初始学习率与原始学习率的比例;     | 
| save_interval         | int  |   模型保存的间隔，即每训练多少个iteration保存一次模型。当训练时间较长时，保存间隔可以避免因突然中断或出现错误导致训练成果全部丢失;   | 
| log_interval           |int  |    日志输出的间隔，即每训练多少个iteration输出一次日志信息    | 
| lora           |int  |    日志输出的间隔，即每训练多少个iteration输出一次日志信息    | 
| enable_sft_dataset_dir           |str  |    SFT训练数据集的目录    | 
| enable_sft_dataset_file           |str  |    SFT训练数据集的文件名    | 

完整参数信息可参考https://github.com/FlagAI-Open/FlagAI/blob/master/flagai/env_args.py

🌟对于推理任务，可在`generate_chat.py`文件里执行`aquila_generate`函数时重设下列参数:

|   参数名          |  类型   | 默认值  | 描述  |                                  
| :---------------- | :------- | :-- |  :-- |   
| temperature       | float  | 0.8  |   温度控制着模型生成新词时的随机性程度。在基于概率的语言模型中，每个词都有一个与之对应的概率分布，温度通过增加或减少这些概率分布来影响模型生成单词的随机性。较高的温度会使得模型更倾向于选择概率较小的单词，从而生成更多的“冒险”文本。相反，较低的温度会强制模型更加倾向于选择概率最大的单词，从而生成更加可预测的文本。常见的温度值范围为0.5-1.5。   |
| topk         |int  | 30  |    Top-k控制着模型生成新词时的选择数量。在生成每个新词时，模型会预测出若干个可能的单词，Top-k参数会限制模型只选择概率最大的前k个单词中的一个来作为生成的单词。Top-k可以帮助稳定生成过程，防止模型随意选择概率很小的单词。     |
| topp        |float  |0.95  |     跟Top-k类似，Top-p也是控制着模型生成新词时的选择数量。在生成每个新词时，模型会预测出若干个可能的单词，Top-p参数会限制模型只选择概率最高的一些候选单词，直到这些候选单词的总概率达到一个阈值（如0.9或0.8）。Top-p可以帮助避免产生不符合语境的单词。    |   
| max_length           |int | 200  |   为了避免生成无限长的文本，我们需要限制生成的文本长度。Max_length参数控制生成文本的最大长度，一旦达到该长度，模型就会停止生成。Aquila系列模型的最大长度为2048个token。   | 

## 证书

AquilaChat-7B和AquilaChat-33B开源模型使用 [智源Aquila系列模型许可协议](../../../BAAI_Aquila_Model_License.pdf), 原始代码基于[Apache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0)。

