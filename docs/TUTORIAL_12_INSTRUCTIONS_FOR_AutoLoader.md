## AutoLoader

### Quickly build models and tokenizers with Autoloader
Autoloader automatically searches for pre-trained models and Tokenizers from the ModelHub based on the **model_name** and downloads them to the **model_dir**.

Take semantic matching tasks as an example:
```python
## Target contains all target categories
## 0 means that two sentences have the same meaning
## 1 means that two sentences have different meanings
target = [0, 1]
auto_loader = AutoLoader(task_name="cls", ## The task name
                         model_name="RoBERTa-wwm-ext", ## The model name.
                         model_dir=model_dir, ## Model download folder
                         load_pretrain_params=True, ## Whether to load the pretraining model parameters. If False, only the model will be built and the pretraining parameters will not be downloaded.
                         target_size=len(target) ## The final output size of model. Use for classification.
                         )
```

### All supported tasks
The **task_name** parameter supports:
1. task_name="cls": Supports a variety of classified tasks, for example, text classification, semantic matching, emotion analysis...
2. task_name="seq2seq": Supports seq2seq tasks, for example, auto title generation, auto couplet, auto chat...
3. task_name="sequence_labeling": Supports sequence labeling tasks, for example, ner, the part of speech tagging, chinese word segmentation...
4. task_name="sequence_labeling_crf": Add conditional random field layer for sequence labeling model.
5. task_name="sequence_labeling_gp": Add global pointer layer for sequence labeling model.

### All supported models
All supported models is can be found in **model hub**.
Different models adapt to different tasks.

#### Transfomrer encoder:

For example, model_name="bert-base-chinese" or "RoBERTa-wwm-ext" These models support all of the tasks mentioned in the previous section

#### Transformer decoder:

For example, model_name="gpt2-chinese", the model support "seq2seq" task.

#### Transformer encoder + decoder:

For example model_name="t5-base-chinese", the model support "seq2seq" task.
