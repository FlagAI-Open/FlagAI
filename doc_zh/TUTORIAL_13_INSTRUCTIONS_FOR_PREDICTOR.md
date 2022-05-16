## Predictor

### 通过Predictor快速得到预测结果
通过Predictor，只需要输入一个**文本**，就可以直接得到对应任务的输出。

以gpt2编写任务为例：
```python
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
if __name__ == '__main__':
    ## 通过autoloader加载模型和tokenizer
    loader = AutoLoader("seq2seq", "gpt2_base_chinese", model_dir="./state_dict/")
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    ## 定义预测器
    predictor = Predictor(model, tokenizer)
    ## 定义输入预计的开头，作为输入
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

### 所有支持的方法
1. predict_cls_classifier: 输入文本或文本对得到多分类结果，支持bert、roberta等transformer编码器模型。
2. predict_masklm: 输入带有[MASK]标记的文本得到原文结果，支持bert、roberta等transformer编码器模型
3. predict_ner: 输入文本得到ner结果，支持bert、roberta等transformer编码器模型。
4. predict_generate_beamsearch: 输入文本得到输出文本，属于seq2seq任务。支持bert、roberta、gpt2、t5和glm模型。
5. predict_generate_randomsample: 输入文本得到输出文本，属于seq2seq任务。支持bert、roberta、gpt2、t5和glm模型。.

