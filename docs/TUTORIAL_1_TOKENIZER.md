# How to construct and use Tokenizer

## What is Tokenization?
**Tokenization** is a fundamental step in the preprocessing stage of NLP projects, 
and its purpose is to convert unstructured symbolic texts into numeric matrices,
which are suitable for machine learning systems.

In the tokenization process, a **tokenizer** is used to split natural language text 
into a sequence of semantic units called **tokens**, which are then converted into
ids by looking up the tokens in a vocabulary file. An example of tokenizing 
input text `Jack is walking a dog.` is shown below:

<div align=center><img src="img/tokenizer_example_1.png" width="500px"></div>

It is noticeable that different tokenizers can have different ways to split text,
and have different vocabulary files. 

[//]: # (An introduction to those algorithms can be viewed [here]&#40;tokenization.md&#41;.)




## Loading a tokenizer
```python
from flagai.data.tokenizer import Tokenizer
model_name = "GLM-large-en"
tokenizer = Tokenizer.from_pretrained(model_name) # Load tokenizer 
```
At this step, the vocab files from Modelhub will be automatically downloaded to the path specified in `cache_dir` parameter. It is set to `./vocab` directory under the tokenizer file in default.  

## Applying a tokenizer
The tokenizer can be used to encode text to a list of token IDs, as well as decoding the token IDs to the original text. 
```python
text = "Jack is walking a dog."                  # Input text
encoded_ids = tokenizer.EncodeAsIds(text)        # Convert text string to a list of token ids
# Now encoded_ids = [2990, 2003, 3788, 1037, 3899, 1012]
recoverd_text = tokenizer.DecodeIds(encoded_ids) # Recover text string
# recovered_text should be the same as text
```
## Creating your own tokenizer
Different tokenizers has different vocabulary and different ways to split text. To suit your project, sometimes it is significant to create a new tokenizer, and how to implement that is given below: 
### 1. Create a package under `/flagai/tokenizer`

### 2. Wrap the tokenizer from huggingface

let's take T5 tokenizer as an example

```python
from transformers import T5Tokenizer
from ..tokenizer import Tokenizer
class T5BPETokenizer(Tokenizer):
    def __init__(self, model_type_or_path="t5-base", cache_dir=None):
        self.text_tokenizer = T5Tokenizer.from_pretrained(model_type_or_path,
                                                            cache_dir=cache_dir)
        self.text_tokenizer.max_len = int(1e12)
```

