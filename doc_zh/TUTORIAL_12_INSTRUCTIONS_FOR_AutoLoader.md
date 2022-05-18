## AutoLoader

### 使用Autoloader快速建立模型以及分词器
Autoloader会根据**model_name**从ModelHub中自动搜索预训练模型和Tokenizer，并将它们下载到**model_dir**



以语义匹配任务为例:
```python
## target包含所有目标分类
## 0 代表两句话的意思相似
## 1 代表两句话有着不同的意思
target = [0, 1]
auto_loader = AutoLoader(task_name="classification", ## 任务名
                         model_name="RoBERTa-base-ch", ## 模型名字
                         model_dir=model_dir, ## 模型下载的目录
                         load_pretrain_params=True, ## 是否要加载已有的预训练模型参数.
                         target_size=len(target) ## 最终输出的维度，用来进行分类任务.
                         )
```

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
