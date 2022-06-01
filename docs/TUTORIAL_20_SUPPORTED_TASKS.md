# All supported tasks

You can input the different "task_name" parameters in AutoLoader to load model to perform different task.

```python
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

auto_loader = AutoLoader(
    "seq2seq",
    model_name="RoBERTa-base-ch",
    load_pretrain_params=True,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
The "task_name" is "seq2seq" in this example. We construct a roberta seq2seq model by the AutoLoader class.

We will list all currently supported tasks below.
### All supported tasks
The **task_name** parameter supports:
1. task_name="classification": Supports a variety of classified tasks, for example, text classification, semantic matching, emotion analysis...
2. task_name="seq2seq": Supports seq2seq tasks, for example, auto title generation, auto couplet, auto chat...
3. task_name="sequence_labeling": Supports sequence labeling tasks, for example, ner, the part of speech tagging, chinese word segmentation...
4. task_name="sequence_labeling_crf": Add conditional random field layer for sequence labeling model.
5. task_name="sequence_labeling_gp": Add global pointer layer for sequence labeling model.

### All supported models
All supported models is can be found in **model hub**.
Different models adapt to different tasks.

#### Transfomrer encoder:

For example, model_name="BERT-base-ch" or "RoBERTa-base-ch" These models support all of the tasks mentioned in the previous section, such as NER(sequence labeling), text classification, semantic matching, seq2seq and so on.

#### Transformer decoder:

For example, model_name="GPT2-base-ch", the model support "seq2seq" task. Input a beginning of a sentence, the model can continue writing.

#### Transformer encoder + decoder:

For example model_name="T5-base-ch", the model support "seq2seq" task.