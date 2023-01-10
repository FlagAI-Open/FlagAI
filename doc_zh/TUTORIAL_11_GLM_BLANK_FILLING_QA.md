# GLM 空白填充

![img.png](./img/glm_blank_filling.png)
GLM 自然地处理可变长度的空白填充，这对于许多下游任务至关重要。
基于自回归的填充，GLM训练的时候会遵循自编码的做法随机从输入文本中删除连续的token，以及
然后遵循自回归的做法训练模型重建这些被删除的部分。

GLM 对下游任务进行微调的时候，无论是何种形式的人物，都会被重新定义为空白填充任务。
例如，在情感分类任务（预测文本蕴含的情绪是积极的还是消极的）中，原始文本后面会添加如下带有空白的被重新表述为填充如下空白。  
"它真的____"， 然后模型去预测填空处为"好"或"坏"，表明
情绪是积极的还是消极的。

在 GLM 中，共有三种遮挡的方法，分别对应三种预测格式，如下所示：
1. ```[MASK]```, 表示遮挡Token级别的文本。这种方法只会遮挡句子中的随机token，
   在三种 MASK 方法中，它的掩码部分最少。所以，生成的内容也有限。例如： ``[CLS]北京故宫是中国[MASK]非物质文化遗产。<|endoftext|><|startofpiece|>现存最大的古代宫殿建筑, 也是``.
2. ```[sMASK]```: 表示遮挡句子级别的文本。我们限制遮挡的区间必须是完整的句子。总长度15%的区间被抽样覆盖。该任务目标旨在执行seq2seq任务，其预测通常是完整的句子或段落。
  ```[sMASK]``` 预测结果比 ```[MASK]```更长. 例如: `` [CLS]人工智能是一个以计算机科学为基础,由计算机、数学、哲学等多学科交叉融合的交叉学科,[sMASK],具有非常巨大的前景。<|endoftext|><|startofpiece|>它涉及的信息量不仅非常巨大,而且也是人工智能发展的一个关键,其研究内容包括人的感觉、知觉和思维,以及如何理解各种现象,以及解释现象的本性和原因等,通过计算机来进行系统的分析推理、建立数学模型,并模拟人类意识。``
3. ```[gMASK]```: 表示遮挡文档级别的文本。我们对单个较长的跨度进行采样，而其长度是从原始长度的 50%–100% 上的均匀分布中采样的。
   该目标旨在生成长文本。 例如: ``[CLS]问题:啤酒伤胃吗?回答:[gMASK]<|startofpiece|>谢邀。 我是啤酒爱好者,但是我不喝酒。 我以前也说过,喝酒伤身,啤酒伤胃,伤肠道。 现在我也知道了啤酒伤人的很多细节,我就不瞎说,大家看图片就知道了。 <n><n>其实啤酒伤身这个说法只是表面而已。 <n><n>啤酒中含有少量的碳酸和酒精,碳酸和酒精是成酸性物质,而乙醇是脂溶性的,酒精在胃里能够被分解,生成乙醇和二氧化碳,在体内是水和二氧化碳,两种物质会迅速发生中和反应,结果导致人体出现头痛、呕吐、胸痛、浑身发热等现象,这就是所谓喝大了,喝多了。 <n><n> 啤酒的含糖量在15%左右,喝多了也是伤身的,啤酒含糖量较高的主要成分是水分,而水分的体积比酒精大,所以酒精进入人体,与水相遇,就会产生大量气体,二氧化碳、水、一氧化碳等刺激人体,造成人体大量出汗,使体内温度升高,``


例如，我们用`GLM-large-ch`模型以自回归的形式完成空白填充任务，如果想要使用更大的百亿参数模型`GLM-10b-ch`请点[这里](https://model.baai.ac.cn/model-detail/100001):
```python
import torch
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import Tokenizer
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples') 
    model_name = 'GLM-large-ch'
    model = GLMModel.from_pretrain(model_name=model_name,
                                   download_path="./state_dict/")
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer = Tokenizer.from_pretrained(model_name, only_download_config=False)
    model.cuda(torch.cuda.current_device())
    predictor = Predictor(model, tokenizer)
    # question-answering
    text = '问题：啤酒伤胃吗？回答：[gMASK]'
    output=predictor.predict_generate_randomsample(text)
    print(text,'\n',output)
```

与 BERT 类似，GLM 可以进行被遮挡token的预测:
```python
import torch
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import Tokenizer
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples') 
    model_name = 'GLM-large-ch'
    tokenizer = Tokenizer.from_pretrained(model_name)
    model = GLMModel.from_pretrain(model_name=model_name, only_download_config=False)
    model.cuda(torch.cuda.current_device())
    predictor = Predictor(model, tokenizer)
    # question-answering
    text = '北京故宫是中国[MASK]非物质文化遗产。'
    output=predictor.predict_generate_randomsample(text)
    print(text,'\n',output)
```
以及被遮挡句子的预测:

```python
import torch
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples') 
    model_name = 'GLM-large-ch'
    tokenizer = Tokenizer.from_pretrained(model_name)
    model = GLMModel.from_pretrain(model_name=model_name, only_download_config=False)
    model.cuda(torch.cuda.current_device())
    predictor = Predictor(model, tokenizer)
    # question-answering
    text = '人工智能是一个以计算机科学为基础，由计算机、数学、哲学等多学科交叉融合的交叉学科，[sMASK]，具有非常巨大的前景。'
    output=predictor.predict_generate_randomsample(text)
    print(text,'\n',output)
```