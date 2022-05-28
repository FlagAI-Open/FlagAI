## AutoLoader

### 通过 AutoLoader 快速构建支持不同下游任务的大型模型。
![](./img/autoloader_map.png)

AutoLoader 可以快速找到对应的预训练模型和分词器，只需输入 task_name 和 model_name 即可。

以标题生成任务为例：

```python
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(task_name="title-generation", ## The task name
                         model_name="RoBERTa-base-ch", ## The model name.
                         )
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
```task_name``` 你要想做的任务名字, 除了 ```title-generation```, 你也可以选择其他的细分任务, 例如 ```semantic-matching```、```ner``` and so on.
```python
target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(task_name="ner",
                         model_name="RoBERTa-base-ch",
                         class_num=len(target))
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
在构建分类相关任务的模型时，还需要输入一个“class_num”参数来告诉模型要分类多少类。

然后 AutoLoader 将从模型中心下载 roberta 预训练模型、配置和词表。
并且下载的预训练模型、配置和词表将被放入“./state_dict/RoBERTa-base-ch”目录中。

另外，通过输入不同的model_name，也可以直接调用已经训练好的下游任务模型，如``Roberta-base-ch-ner``、``Roberta-base-ch-title-generation``等。
```python
target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(task_name="ner",
                         model_name="RoBERTa-base-ch-ner",
                         class_num=len(target))

model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
查看 https://github.com/BAAI-Open/FlagAI/blob/master/quickstart/ner_ch.py 了解更多.
## All supported tasks and models

![model_and_task_table](../docs/img/model_task_table.png)
 **task_name** 参数支持:
### 分类任务
1. classification
2. semantic-matching
3. emotion-analysis (todo)
4. text-classification (todo)
5. ...

### 生成任务
1. seq2seq
2. title-generation
3. writing
4. poetry-generation
5. couplets-generation (todo)
6. ...
### 序列标注任务
1. sequence-labeling
2. ner
3. ner-crf
4. ner-gp
5. part-speech-tagging (todo)
6. chinese-word-segmentation (todo)
7. ...

## 所有支持的模型
所有支持的模型都可以在 **model hub** 中找到。
不同的模型适应不同的任务。

#### Transfomrer encoder:

例如，model_name="GLM-large-ch" 或 "RoBERTa-base-ch" 这些模型支持上一节中提到的所有任务。

#### Transformer decoder:

例如，model_name="GPT2-base-ch"，模型支持“seq2seq”相关任务。

#### Transformer encoder + decoder:

例如model_name="t5-base-ch"，模型支持“seq2seq”相关任务。