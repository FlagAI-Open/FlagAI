# 所有支持的任务


您可以在 AutoLoader 中输入不同的“task_name”参数来加载模型以执行不同的任务。

```python
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

auto_loader = AutoLoader(
    "seq2seq",
    model_name="RoBERTa-base-ch",
    load_pretrain_params=True,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
在这个例子中，“task_name”是“seq2seq”。我们通过AutoLoader类构建了一个roberta seq2seq 模型。

我们将在下面列出所有当前支持的任务。
### 所有支持的任务
**task_name**参数可以为如下值:
1. task_name="classification": 支持不同的分类任务，例如文本分类， 语义匹配， 情感分析...
2. task_name="seq2seq": 支持序列到序列的模型, 例如标题自动生成, 对联自动生成, 自动对话...
3. task_name="sequence_labeling": 支持序列标注任务， 比如实体检测，词性标注，中文分词任务...
4. task_name="sequence_labeling_crf": 为序列标注模型添加条件随机场层.
5. task_name="sequence_labeling_gp": 为序列标注模型添加全局指针层.

### 所有支持的模型
所有支持的模型都可以在 **model hub** 中找到。
不同的模型适应不同的任务。

#### Transfomrer编码器:

例如 model_name="BERT-base-ch" or "RoBERTa-base-ch"时， 这些模型支持上一节中提到的所有任务

#### Transformer解码器:

例如 model_name="GPT2-base-ch"时, 模型支持 "seq2seq" 任务.

#### Transformer 编码器+解码器:

例如 model_name="T5-base-ch"时, 模型支持"seq2seq" task.