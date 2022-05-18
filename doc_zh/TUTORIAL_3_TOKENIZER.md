# 分词器

## 支持的分词器列表
| 分词器                         | 语言  | 相关模型(參考 [ModelHub](https://model.baai.ac.cn/models)) |
|-----------------------------|-----|------------------------------------------------------|
| GLMLargeEnWordPieceTokenizer | 英文  | GLM-large-en                                         |
| GLMLargeChTokenizer         | 中文  | GLM-large-ch                                         |
| GLM10bENBPETokenizer        | 英文  | glm_10b_en                                           |
| T5BPETokenizer              | 中文  | t5_base                                              |
| ROBERTATokenizer            | 中文  | RoBERT-base-ch                                       |
| BertWordPieceTokenizer      | 中文  |                                                      |

## 介绍

自然语言通常以一连串符号的形式表示，我们将每个符号称为字符。

在自然语言文本里，每个语义单元(我们称之为token)都由一个或几个连续的字符组成。例如，“狐狸”，
作为一种杂食性的犬科动物，表示为“狐”和“狸”。
将文本拆分为一连串语义单元的过程被定义为分词(Tokenization)，下面展示了一个例子：


原始语句:                   小明在北京上学。

分词后的语句:      [小明,   在,   北京,   上学,   。]

需要注意的是切割的方法可以自由选择。语义单元可以是单字，也可以是更长的文本，只要其包含适量的语义信息就好。

另一个关键的预处理步骤是矢量化，它将原始token序列转换为数字矢量或矩阵，以便可以直接输入到我们的语言模型中。通常会使用一个字典文件将每个token映射到其相应的序号。

在我们的项目中，有一些进行分词的类，我们称为分词器。其中每个分词器都能以不同的方式对原始文本进行分词和矢量化。

## 加载分词器
```python
from flagai.data.tokenizer import GLMLargeEnWordPieceTokenizer
tokenizer = GLMLargeEnWordPieceTokenizer()
```


## 创建分词器
想创建新的分词器的时候, 需要这么做:
### 1. 在`/flagai/tokenizer`目录下建立一个新的目录

### 2. 在目录下新建一个python文件，在里面添加自定义的分词器

让我们以T5为例，实现一下自定义的分词器

```python
from transformers import T5Tokenizer
from ..tokenizer import Tokenizer
class T5BPETokenizer(Tokenizer):
    def __init__(self, model_type_or_path="t5-base", cache_dir=None):
        self.text_tokenizer = T5Tokenizer.from_pretrained(model_type_or_path,
                                                            cache_dir=cache_dir)
        self.text_tokenizer.max_len = int(1e12)
```
如果`model_type_or_path`这项参数的值已经指向了一个huggingface transformers里的分词器，那现在已经成功自定义分词器了！

否则，您需要自己在此类下方实现以下功能。


```python
def EncodeAsIds(self, text: str, process_fn=None):
    """输入文本 => 一个token序号列表"""

def EncodeAsTokens(self, text: str, process_fn=None):
    """输入文本 => 一个token列表"""

def IdToToken(self, Id: int):
    """Token序号 => token"""

def TokenToId(self, token: str):
    """Token => token序号"""
    return self.text_tokenizer._convert_token_to_id(token)

def DecodeIds(self, Ids: list[int]):
    """一个token序号列表 => 对应的文本"""
    return self.DecodeTokens([self.IdToToken(id) for id in Ids])

def DecodeTokens(self, tokens: list[str]):
    """一个token列表 => 对应的文本"""
    return self.text_tokenizer.convert_tokens_to_string(tokens)
```