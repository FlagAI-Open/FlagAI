GLM  naturally handles variable-length blank filling which is crucial for many downstream tasks.  
Based on autoregressive blank-filling, GLM randomly
blank out continuous spans of tokens from the input text,
following the idea of autoencoding, and train the model to
reconstruct the spans, following the idea of autoregressive
pre-training. 
As a result, GLM learns 
autoregressive generation during pre-training.


GLM finetunes   on downstream tasks and reformulate them as blank-filling generation. Each task is associated with a human-crafted cloze question, and the model predicts the answer
to the cloze. For example, a sentiment classification task
is reformulated as filling the blank in “[SENTENCE]. It’s
really ”. The prediction of “good” or “bad” indicates the
sentiment being positive or negative. With such formulation,
GLM benefits from the consistency between pretraining and
finetuning, because both pretraining and finetuning involves
training the model to generate text given context.   To make the pre-training
method better suited for text generation tasks, GLM also studys
a multi-task pre-training setup, where the model is jointly
trained to reconstruct masked spans and generate longer
text.

In GLM, there are three MASK methods, corresponding to three prediction formats respectively.
1. ```[MASK]```, Token-level. This mask method will only mask random tokens in a sentence, 
   which has the least mask part among the three maks methods. Therefore, 
   the content generated is also limited. For example: ``[CLS]北京故宫是中国[MASK]非物质文化遗产。<|endoftext|><|startofpiece|>现存最大的古代宫殿建筑, 也是``.
2. ```[sMASK]```: Entity-level. We restrict that the masked
   spans must be full sentences. Multiple spans
   (sentences) are sampled to cover 15% of
   the original tokens. This objective aims for
   seq2seq tasks whose predictions are often
   complete sentences or paragraphs. 
   The ```[sMASK]``` is longer than the text predicted by ```[MASK]```. For example: `` [CLS]人工智能是一个以计算机科学为基础,由计算机、数学、哲学等多学科交叉融合的交叉学科,[sMASK],具有非常巨大的前景。<|endoftext|><|startofpiece|>它涉及的信息量不仅非常巨大,而且也是人工智能发展的一个关键,其研究内容包括人的感觉、知觉和思维,以及如何理解各种现象,以及解释现象的本性和原因等,通过计算机来进行系统的分析推理、建立数学模型,并模拟人类意识。``
3. ```[gMASK]```: Document-level. We sample a single span
   whose length is sampled from a uniform distribution over 50%–100% of the original length.
   The objective aims for long text generation. For example: ``[CLS]问题:啤酒伤胃吗?回答:[gMASK]<|startofpiece|>谢邀。 我是啤酒爱好者,但是我不喝酒。 我以前也说过,喝酒伤身,啤酒伤胃,伤肠道。 现在我也知道了啤酒伤人的很多细节,我就不瞎说,大家看图片就知道了。 <n><n>其实啤酒伤身这个说法只是表面而已。 <n><n>啤酒中含有少量的碳酸和酒精,碳酸和酒精是成酸性物质,而乙醇是脂溶性的,酒精在胃里能够被分解,生成乙醇和二氧化碳,在体内是水和二氧化碳,两种物质会迅速发生中和反应,结果导致人体出现头痛、呕吐、胸痛、浑身发热等现象,这就是所谓喝大了,喝多了。 <n><n> 啤酒的含糖量在15%左右,喝多了也是伤身的,啤酒含糖量较高的主要成分是水分,而水分的体积比酒精大,所以酒精进入人体,与水相遇,就会产生大量气体,二氧化碳、水、一氧化碳等刺激人体,造成人体大量出汗,使体内温度升高,``


As example, GLM finish the question task as an autoregressive blank in-
filling task

```python
import torch
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples') 
    tokenizer = GLMLargeChTokenizer(vocab_path='./checkpoints/glm-large-ch/cog-pretrain.model',
                                    add_block_symbols=True,
                                    add_task_mask=True,
                                    add_decoder_mask=False,
                                    fix_command_token=False)
    model = GLMModel.from_pretrain(model_name='glm-large-ch', only_download_config=False)
    model.cuda(torch.cuda.current_device())
    predictor = Predictor(model, tokenizer)
    # question-answering
    text = '问题：啤酒伤胃吗？回答：[gMASK]'
    output=predictor.predict_generate_randomsample(text)
    print(text,'\n',output)
```

Similar to BERT, GLM can predict masked tokens as 

```python
import torch
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples') 
    tokenizer = GLMLargeChTokenizer(vocab_path='./checkpoints/glm-large-ch/cog-pretrain.model',
                                    add_block_symbols=True,
                                    add_task_mask=True,
                                    add_decoder_mask=False,
                                    fix_command_token=False)
    model = GLMModel.from_pretrain(model_name='glm-large-ch', only_download_config=False)
    model.cuda(torch.cuda.current_device())
    predictor = Predictor(model, tokenizer)
    # question-answering
    text = '北京故宫是中国[MASK]非物质文化遗产。'
    output=predictor.predict_generate_randomsample(text)
    print(text,'\n',output)
```
and predict masked sentences as 

```python
import torch
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples') 
    tokenizer = GLMLargeChTokenizer(vocab_path='./checkpoints/glm-large-ch/cog-pretrain.model',
                                    add_block_symbols=True,
                                    add_task_mask=True,
                                    add_decoder_mask=False,
                                    fix_command_token=False)
    model = GLMModel.from_pretrain(model_name='glm-large-ch', only_download_config=False)
    model.cuda(torch.cuda.current_device())
    predictor = Predictor(model, tokenizer)
    # question-answering
    text = '人工智能是一个以计算机科学为基础，由计算机、数学、哲学等多学科交叉融合的交叉学科，[sMASK]，具有非常巨大的前景。'
    output=predictor.predict_generate_randomsample(text)
    print(text,'\n',output)
```