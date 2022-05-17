# Tokenizer

## Supported tokenizers
| Tokenizer                    | Language | Related model (see https://model.baai.ac.cn/models) |
|------------------------------|----------|-----------------------------------------------------|
| GLMLargeEnWordPieceTokenizer | English  | GLM-large-en                                        |
| GLMLargeChTokenizer          | Chinese  | GLM-large-ch                                        |
| GLM10bENBPETokenizer         | English  | glm_10b_en                                          |
| T5BPETokenizer               | Chinese  | t5_base                                             |
| ROBERTATokenizer             | Chinese  | RoBERTa-wwm-ext                                     |
| BertWordPieceTokenizer       | Chinese  |                                                     |

## Introduction

自然语言通常以一连串符号的形式表示，我们将每个符号称为字符。

在任何文本中，每个语义单元都由几个连续的字符组成，我们将每个语义单元称为一个令牌。例如，“狐狸”，
一种杂食性的犬科动物，表示为“狐”和“狸”。
将文本拆分为标记序列的过程被定义为标记化，下面显示了一个示例：


Original sentence:                   Jack is walking a dog.

Tokenized sentence:    [Jack,   is,   walking,   a,   dog,    .]

It is noticeable that token is not equivalent to word. It can also be character, sub-word, sentence piece and so on as long as it holds appropriate amount of semantic information.

Another critical preprocessing step is vectorization,  which turns the raw symbolic sequences into a numeric vector or matrix so that it can be directly fed into our language model. Usually there is achieved by using a vocabulary file to map each token to its corresponding id.

In our project, there are a bunch of tokenizer classes, where each of them can tokenize and vectorize raw texts in different ways, and there are also other important functions.

## Loading tokenizer
Load an existing tokenizer:
```python
from flagai.data.tokenizer import GLMLargeEnWordPieceTokenizer
tokenizer = GLMLargeEnWordPieceTokenizer()
```

## Creating tokenizer
To create a new tokenizer, you need to:
### 1. Create a package under /flagai/tokenizer

### 2. create a python file to define the tokenizer

Initialize the tokenizer as below (let's take T5 tokenizer as an example)

```python
from transformers import T5Tokenizer
from ..tokenizer import Tokenizer
class T5BPETokenizer(Tokenizer):
    def __init__(self, model_type_or_path="t5-base", cache_dir=None):
        self.text_tokenizer = T5Tokenizer.from_pretrained(model_type_or_path,
                                                            cache_dir=cache_dir)
        self.text_tokenizer.max_len = int(1e12)
```
If tokenizer model imported from transformers is used as the text tokenizer, it is all done!

Otherwise, you need to implement the following class functions by your own.


```python
def EncodeAsIds(self, text: str, process_fn=None):
    """Input text string => a list of token ids"""

def EncodeAsTokens(self, text: str, process_fn=None):
    """Input text string => a list of tokens"""

def IdToToken(self, Id: int):
    """Token id => token"""

def TokenToId(self, token: str):
    """Token => token id"""
    return self.text_tokenizer._convert_token_to_id(token)

def DecodeIds(self, Ids: list[int]):
    """A list of token ids => recovered text string"""
    return self.DecodeTokens([self.IdToToken(id) for id in Ids])

def DecodeTokens(self, tokens: list[str]):
    """A list of tokens => recovered text string"""
    return self.text_tokenizer.convert_tokens_to_string(tokens)
```