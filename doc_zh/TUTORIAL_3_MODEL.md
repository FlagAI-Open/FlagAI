# 模型的主要功能及相关结构

基类 BaseModel 实现了从本地文件或目录或从库提供的预训练模型配置（从BAAI modelhub 的 金山 S3 存储库下载）加载/保存模型的常用方法。
现在所有支持的模型，对三种最常见的模型类型【encoder，decoder和encoder-decoder】进行了支持。现在GLM模型可以加载所有GLM系列的模型，详见 https://github.com/THUDM/GLM

## From_pretrain

同一个模型结构的模型可以用同一个class进行加载，比如BERT-base 和Roberta-base模型都能用BertModel这个class进行加载。From_pretrain为了数据/模型并行的模型加载进行了特定优化，避免重复下载导致的资源浪费。
通过调用ClassName.from_pretrian()来进行加载，现在我们的model hub中对以下的模型进行了支持，可以直接下载模型配置文件【config.json】，模型权重[pytorch_model.bin]，以及字典文件[vocab.txt]。例子：
```python
from flagai.model.glm_model import GLMForSingleTokenCloze
model = GLMForSingleTokenCloze.from_pretrain(download_path="./state_dict", model_name="GLM-large-ch")
```
如果是从本地加载模型权重，也可以通过 ClassName.from_pretrain()进行加载。例子：
从`./state_dict/GLM-large-ch`目录中加载模型文件 `pytorch_model.bin`
```python
from flagai.model.glm_model import GLMForSingleTokenCloze
model = GLMForSingleTokenCloze.from_pretrain(download_path="./state_dict",
                                               model_name="GLM-large-ch")
```
## 所有支持模型

| ClassName                         | ModelName       | Language | Model Type |
|-----------------------------------|-----------------|----------|------------|
| flagai.model.glm_model.GLMModel   | GLM-10b-ch      | chinese  | encoder    |
| flagai.model.glm_model.GLMModel   | GLM-large-ch    | chinese  | encoder    |
| flagai.model.bert_model.BertModel | RoBERTa-base-ch | chinese  | encoder    |
| flagai.model.gpt2_model.GPT2Model | GPT2_base_ch    | chinese  | decoder    |
| flagai.model.t5_model.T5Model     | T5-base-ch      | chinese  | enc2dec    |
| flagai.model.t5_model.T5Model     | T5-base-en      | chinese  | enc2dec    |
| flagai.model.bert_model.BertModel | BERT-base-en    | english  | encoder    |
| flagai.model.glm_model.GLMModel   | GLM-large-en    | english  | encoder    |

## 支持的模型+任务

同时，我们对在任务上finetune好的模型进行了支持，如下表所示，可以通过ClassName.from_pretrain()来加载模型权重，例如，我们自动下载并加载一个在title-generation任务上训练好的GLM-large-ch模型：
```python
from flagai.model.glm_model import GLMForSeq2Seq
model = GLMForSeq2Seq.from_pretrain(model_name='GLM-large-ch')
```
我们也提供了AutoLoader类来帮助加载模型，比如GLM-large-ch模型用于seq2seq任务，这里我们采用了任务和模型独立的设计，理论上任务和模型可以自由更换。
```python
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader("seq2seq",
                         model_name="GLM-large-ch",
                         model_dir= "./state_dict")
model = auto_loader.get_model()
```
| ClassName                                       | Model Name      | language | Task              |
|-------------------------------------------------|-----------------|----------|-------------------|
| flagai.model.glm_model.GLMForSeq2Seq            | GLM-large-ch    | chinese  | title generation  |
| flagai.model.glm_model.GLMForSeq2Seq            | GLM-large-ch    | chinese  | poetry generation |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese  | title generation  |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese  | NER               |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese  | semantic matching |
| flagai.model.t5_model.T5Model                   | T5-base-ch      | chinese  | title generation  |
| flagai.model.bert_model.BertForSequenceLabeling | BERT-base-en    | english  | title gneration   |

## 模型设计
模型主要的构建逻辑`layer->block>model`
`flagai.model.layer`: 包括mlp，layernorm, activation，attention等各种layer层

`flagai.model.block`:通过组装各种layer来构建transformer block，比如BERT block等

`flagai.model`: 通过embedding层和stacked blocks 来构建model

## forward 函数
Model 的forward函数：
输入是 keyword arguments：包括 input_ids, position_ids, attention_mask等，对冗余的参数会自动忽略
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
输出是 dictionary，包括 logits 和hidden states，这两个是必须的，例如GLM forword函数的返回：
```python
return {'loss': loss, 'logits': logits, 'hidden_states': mems}
```
## init_from_json
Model 的init_from json函数：
输入是一个dictionary， 输出是一个初始化的model
例如GLMModel的调用如下：
```python
GLMModel.init_from_json(config_file = "./config.json", **kwargs)
```
**kwargs是预留参数，为了兼容一些模型新增的初始化参数