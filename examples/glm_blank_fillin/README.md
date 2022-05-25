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
1. ```[MASK]```, ```[sMASK]```: Sentence-level. We restrict that the masked
   spans must be full sentences. Multiple spans
   (sentences) are sampled to cover 15% of
   the original tokens. This objective aims for
   seq2seq tasks whose predictions are often
   complete sentences or paragraphs. 
   The ```[MASK]``` is slightly shorter than the text predicted by ```[sMASK]```.
2. ```[gMASK]```: Document-level. We sample a single span
   whose length is sampled from a uniform distribution over 50%–100% of the original length.
   The objective aims for long text generation.


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