# GPT2 generation

## 背景
The GPT2 generation task is to only input the beginning of a text, and the model will continue to write.


![gpt2.png](./img/gpt2_writing_model.png)

## Result show

#### Input
```python
text = "今天天气不错，"
```
#### Output
```
我 们 三 个 女 孩 子 去 吃 的 ， 因 为 是 中 午 ， 所 以 没 有 等 位 。 店 里 面 环 境 还 不 错 ， 比 较 安 静 ， 我 们 要 的 鸳 鸯 锅 底 ， 味 道 一 般 ， 辣 的 感 觉 有 点 不 太 习 惯 。 服 务 态 度 很 好 ， 可 以 刷 卡 。

```
## Usage

### 1. Load model and tokenizer

```python
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
loader = AutoLoader("seq2seq",
                    "GPT2-base-ch",
                    model_dir="./state_dict/")
model = loader.get_model()
tokenizer = loader.get_tokenizer()
predictor = Predictor(model, tokenizer)
```

### Run

```commandline
python ./generate.py
```
Then you can view the running result.