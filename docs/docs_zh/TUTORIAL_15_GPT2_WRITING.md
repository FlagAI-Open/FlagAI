# GPT2 模型生成任务

## 背景
GPT2生成任务是只输入一段文字的开头，模型进行续写



![gpt2.png](../img/gpt2_writing_model.png)

## 结果展示

#### 输入
```python
text = "今天天气不错，"
```
#### 输出
```
我 们 三 个 女 孩 子 去 吃 的 ， 因 为 是 中 午 ， 所 以 没 有 等 位 。 店 里 面 环 境 还 不 错 ， 比 较 安 静 ， 我 们 要 的 鸳 鸯 锅 底 ， 味 道 一 般 ， 辣 的 感 觉 有 点 不 太 习 惯 。 服 务 态 度 很 好 ， 可 以 刷 卡 。

```
## 使用

### 1.模型与切词器加载

```python
from flash_tran.auto_model.auto_loader import AutoLoader
from flash_tran.model.predictor.predictor import Predictor
loader = AutoLoader("seq2seq",
                    "gpt2_base_chinese",
                    model_dir="./state_dict/")
model = loader.get_model()
tokenizer = loader.get_tokenizer()
predictor = Predictor(model, tokenizer)
```

### 运行

```commandline
python ./generate.py
```
然后可以查看运行结果。
