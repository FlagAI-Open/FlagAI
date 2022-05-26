## Predictor

### 通过Predictor调用一行代码得到理想的预测结果

对于不同任务，例如标题生成，命名实体识别，文本分类等，预测的方式各不相同。同时，对于不同的模型，例如encoder、decoder、encoder-decoder模型，预测方式也各不相同。

Predictor中集成了不同任务，不同模型的预测代码，通过Pipline的方式传入文本，Predictor会快速解析模型类型，调用不同的预测代码，得到对应模型的预测结果。

![](./img/predictor_map.png)

以gpt2文章续写任务为例，采用随机采样的生成方式：
```python
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import torch 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 通过autoloader加载模型和tokenizer
    loader = AutoLoader(task_name="writing", 
                        model_name="GPT2-base-ch")
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    model.to(device)
    ## 定义预测器
    predictor = Predictor(model, tokenizer)
    ## 定义文章的开头，作为输入
    text = "今天天气不错，"
    ## 预测器能通过自动分析模型种类来调取不同方法
    out = predictor.predict_generate_randomsample(text,  ## 输入
                                                  input_max_length=512,  ## 最大出入长度
                                                  out_max_length=100, ## 最大输出长度
                                                  repetition_penalty=1.5, ## 避免重复输出. (https://arxiv.org/pdf/1909.05858.pdf)
                                                  top_k=20,  ## 只保留概率最大的k个token.
                                                  top_p=0.8) ## 保留累计概率大于等于top_p的token.(http://arxiv.org/abs/1904.09751)

    print(f"out is {out}")
    ### out is  到这里来看了一下，很是兴奋，就和朋友一起来这里来了。我们是周五晚上去的，人不多，所以没有排队，而且这里的环境真的很好，在这里享受美食真的很舒服，我们点了一个套餐，两个人吃刚刚好，味道很好。
```
Predictor可以自动分析出模型类型为GPT2，并且自动调用GPT2模型对应的生成方法，得到预测结果。

除了```writing```任务之外，Predictor还支持```title-generation```, ``ner``, ``semantic-matching``等任务的预测。例如ner任务：

```python
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
auto_loader = AutoLoader(task_name="ner",
                         model_name="RoBERTa-base-ch-ner", # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
                         class_num=len(target))

model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
model.to(device)

predictor = Predictor(model, tokenizer)

test_data = [
    "6月15日，河南省文物考古研究所曹操高陵文物队公开发表声明承认：“从来没有说过出土的珠子是墓主人的",
    "4月8日，北京冬奥会、冬残奥会总结表彰大会在人民大会堂隆重举行。习近平总书记出席大会并发表重要讲话。在讲话中，总书记充分肯定了北京冬奥会、冬残奥会取得的优异成绩，全面回顾了7年筹办备赛的不凡历程，深入总结了筹备举办北京冬奥会、冬残奥会的宝贵经验，深刻阐释了北京冬奥精神，对运用好冬奥遗产推动高质量发展提出明确要求。",
    "当地时间8日，欧盟委员会表示，欧盟各成员国政府现已冻结共计约300亿欧元与俄罗斯寡头及其他被制裁的俄方人员有关的资产。",
    "这一盘口状态下英国必发公司亚洲盘交易数据显示博洛尼亚热。而从欧赔投注看，也是主队热。巴勒莫两连败，",
]

for t in test_data:
    entities = predictor.predict_ner(t, target, maxlen=256)
    result = {}
    for e in entities:
        if e[2] not in result:
            result[e[2]] = [t[e[0]:e[1] + 1]]
        else:
            result[e[2]].append(t[e[0]:e[1] + 1])
    print(f"result is {result}")
```
通过传入一行文本，快速得到对应的命名实体识别任务结果，并且``predict_ner`` 接口适配所有支持ner任务的模型，例如BERT, Roberta, BERT-CRF, BERT-GlobalPointer, Roberta-CRF, Roberta-GlobalPointer 等等。

![predictor-table](../docs/img/predictor_table.png)


### Perdictor所有支持的方法
#### 文本表征
1. predict_embedding: 输入一个文本来获取嵌入表征，支持bert、roberta等模型。
#### 文本分类，语义匹配
1. predict_cls_classifier: 输入文本或文本对得到多分类结果，支持bert、roberta等transformer编码器模型。
#### Mask语言模型
1. predict_masklm: 输入带有[MASK]标记的文本得到原文结果，支持bert、roberta等transformer编码器模型
#### 命名实体识别
1. predict_ner: 输入文本得到ner结果，支持bert、roberta等transformer编码器模型。
#### 生成
1. predict_generate_beamsearch: 输入文本得到输出文本，属于seq2seq任务。支持bert、roberta、gpt2、t5和glm模型。
2. predict_generate_randomsample: 输入文本得到输出文本，属于seq2seq任务。支持bert、roberta、gpt2、t5和glm模型。

### 方法调用说明
只要所使用的模型支持Predictor的对应方法，便可以直接预测，例如：模型为GLM、T5、GPT2，因为便可以直接调用不同生成方法，但是无法调用其他例如文本分类，命名实体识别预测方法。

Bert、Roberta模型支持的方法较多，可以调用上部分所展示的所有方法。