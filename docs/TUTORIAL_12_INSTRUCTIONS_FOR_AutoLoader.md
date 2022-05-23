## AutoLoader

### Quickly build large models that support different downstream tasks through AutoLoader.

![](./img/autoloader_en_map.png)

AutoLoader can quickly find the corresponding pre-trained model and tokenizer, just enter task_name and model_name.

Take title generation tasks as an example:
```python
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(task_name="title-generation", ## The task name
                         model_name="RoBERTa-base-ch", ## The model name.
                         )
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
```task_name``` is the name of the task you want to do, in addition to ```title-generation```, you can also choose other subdivided tasks, such as ```semantic-matching```、```ner``` and so on.
```python
target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(task_name="ner",
                         model_name="RoBERTa-base-ch",
                         class_num=len(target))
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
When building the model of classification related task, you also need to input a ``class_num`` parameter to tell the model how many classed to classify.

Then AutoLoader will download the roberta pretrained model, config and vocab from model hub.
And the downloaded pretrained model, config and vocab will be put into the "./state_dict/RoBERTa-base-ch" directory.

In addition, by entering a different model_name, you can also directly call the already trained downstream task model, such as ``Roberta-base-ch-ner``, ``Roberta-base-ch-title-generation`` and so on.

```python
target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader(task_name="ner",
                         model_name="RoBERTa-base-ch-ner",
                         class_num=len(target))

model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
See https://github.com/BAAI-Open/FlagAI/blob/master/quickstart/ner_ch.py to learn more.
## All supported tasks and models

![model_and_task_table](./img/model_task_table.png)
The **task_name** parameter supports:
### classification
1. classification
2. semantic-matching
3. emotion-analysis (todo)
4. text-classification (todo)
5. ...

### seq2seq
1. seq2seq
2. title-generation
3. writing
4. poetry-generation
5. couplets-generation (todo)
6. ...
### sequence labeling
1. sequence-labeling
2. ner
3. ner-crf
4. ner-gp
5. part-speech-tagging (todo)
6. chinese-word-segmentation (todo)
7. ...

## All supported models
All supported models is can be found in **model hub**.
Different models adapt to different tasks.

#### Transfomrer encoder:

For example, model_name="GLM-large-ch" or "RoBERTa-base-ch" These models support all of the tasks mentioned in the previous section.

#### Transformer decoder:

For example, model_name="GPT2-base-ch", the model support "seq2seq" related task.

#### Transformer encoder + decoder:

For example model_name="t5-base-ch", the model support "seq2seq" related task.
