# 模型的主要功能及相关结构
- [模型的主要功能及相关结构](#模型的主要功能及相关结构)
  - [基类](#基类)
  - [From_pretrain](#from_pretrain)
    - [从modelhub加载](#从modelhub加载)
    - [从本地加载](#从本地加载)
  - [所有支持模型](#所有支持模型)
  - [支持的模型+任务](#支持的模型任务)
  - [模型设计](#模型设计)
  - [模型的forward 函数](#模型的forward-函数)
    - [输入](#输入)
    - [输出](#输出)
  - [模型的init_from_json函数](#模型的init_from_json函数)
    - [输入](#输入-1)
    - [输出](#输出-1)
## 基类

基类 BaseModel 实现了从本地文件或目录或从库提供的预训练模型配置（从BAAI modelhub 的 金山 S3 存储库下载）加载/保存模型的常用方法。现在对三种最常见的模型类型`encoder`，`decoder`和`encoder-decoder`进行了支持。同一种结构的模型，可以用同一个`Class`来进行加载。比如,`GLMModel`可以加载所有`GLM`系列的模型，详见 https://github.com/THUDM/GLM。

## From_pretrain

`From_pretrain` 函数用于加载模型。同一个模型结构的模型可以用同一个class进行加载，比如`BERT-base-ch` 和`Roberta-base-ch`模型都能用`BertModel`这个`Class`进行加载。`From_pretrain`为了数据/模型并行的模型加载进行了特定优化，避免重复下载导致的资源浪费。
通过调用`ClassName.from_pretrain()`来进行加载.
### 从modelhub加载
现在我们支持从modelhub中下载[常用模型](#所有支持模型)，可以直接通过`from_pretrain`下载模型配置文件`config.json`，模型权重`pytorch_model.bin`，以及字典文件`vocab.txt`。例子：
```python
#从modelhub下载GLM-large-ch 模型
from flagai.model.glm_model import GLMModel
model = GLMModel.from_pretrain(download_path="./state_dict", model_name="GLM-large-ch")
```
### 从本地加载
如果想从本地加载模型权重，也可以通过 `ClassName.from_pretrain()`进行加载。模型权重保存在`download_path/model_name/`中,其中`model_name`是模型相关所在的目录，而`download_path`是`model_name`所在的目录，例子：

```python
#从`./state_dict/GLM-large-ch`目录中加载模型文件 `pytorch_model.bin`和模型配置文件 `config.json`。
from flagai.model.glm_model import GLMModel
model = GLMModel.from_pretrain(download_path="./state_dict", model_name="GLM-large-ch")
```
## 所有支持模型
这些模型都可以通过`from_pretrain` 从model_hub下载。

| ClassName                         | ModelName           | Language | Model Type |
|-----------------------------------|---------------------|----------|------------|
| flagai.model.glm_model.GLMModel   | **GLM-10b-ch**      | chinese  | encoder    |
| flagai.model.glm_model.GLMModel   | **GLM-large-ch**    | chinese  | encoder    |
| flagai.model.bert_model.BertModel | **RoBERTa-base-ch** | chinese  | encoder    |
| flagai.model.gpt2_model.GPT2Model | **GPT2-base-ch**    | chinese  | decoder    |
| flagai.model.t5_model.T5Model     | **T5-base-ch**      | chinese  | enc2dec    |
| flagai.model.t5_model.T5Model     | **T5-base-en**      | chinese  | enc2dec    |
| flagai.model.bert_model.BertModel | **BERT-base-en**    | english  | encoder    |
| flagai.model.glm_model.GLMModel   | **GLM-large-en**    | english  | encoder    |

## 支持的模型+任务
同时，我们对在任务上finetune好的模型进行了支持，如下表所示，可以通过ClassName.from_pretrain()来加载模型权重，例如，我们自动下载并加载一个在title-generation任务上训练好的GLM-large-ch模型：
```python
from flagai.model.glm_model import GLMForSeq2Seq
model = GLMForSeq2Seq.from_pretrain(model_name='GLM-large-ch')
```
为了简化加载流程，我们也提供了AutoLoader类来帮助加载模型，比如GLM-large-ch模型用于seq2seq任务。由于我们采用了任务和模型独立的设计，理论上任务和模型可以自由更换。
```python
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader("title-generation",
                         model_name="GLM-large-ch",
                         model_dir= "./state_dict")
model = auto_loader.get_model()
```
| ClassName                                       | Model Name      | language | Task              |
|-------------------------------------------------|-----------------|----------|-------------------|
| flagai.model.glm_model.GLMForSeq2Seq            | GLM-large-ch    | chinese  | **title generation**  |
| flagai.model.glm_model.GLMForSeq2Seq            | GLM-large-ch    | chinese  | **poetry generation** |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese  | **title generation**  |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese  | **NER**               |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese  | **semantic matching** |
| flagai.model.t5_model.T5Model                   | T5-base-ch      | chinese  | **title generation**  |
| flagai.model.bert_model.BertForSequenceLabeling | BERT-base-en    | english  | **title gneration**   |

## 模型设计
模型主要的构建逻辑`layer->block->model`

`flagai.model.layer`: 包括mlp，layernorm, activation，attention等各种layer层

`flagai.model.block`:通过组装各种layer来构建transformer block，比如BERT block等

`flagai.model`: 通过embedding层和stacked blocks 来构建model

如果想要自定义一个新的模型结构，可以参考上述构建过程。

## 模型的forward 函数
### 输入
输入都是 keyword arguments：包括 input_ids, position_ids, attention_mask等，对冗余的参数会自动忽略
比如GLM的forward 函数：
```python
def forward(self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            mems=None,
            return_memory=False,
            detach_memory=True,
            prompt_pos=None,
            **kwargs)
```
### 输出
输出是 dictionary，包括 logits 和hidden states，这两个是必须的，例如GLM forword函数的返回：
```python
return {'loss': loss, 'logits': logits, 'hidden_states': mems}
```
## 模型的init_from_json函数
### 输入
输入是一个`json`，以及预留参数 `**kwags`，为了兼容一些模型新增的初始化参数

例如GLMModel的调用如下：
```python
GLMModel.init_from_json(config_file = "./config.json", checkpoint_activations=True)
```
`checkpoint_activations=True`是新增的参数，用于控制是否进行梯重计算
### 输出
输出是一个初始化的model