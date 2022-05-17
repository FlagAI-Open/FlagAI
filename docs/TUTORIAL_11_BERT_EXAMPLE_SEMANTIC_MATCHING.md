# Bert example: Semantic matching

## Background

Semantic matching task is to judge whether the meanings of two input sentences are the same, which requires input sentence pairs and output 2 classification results (two sentences have the same or different meanings).

![semantic_model.png](./img/semantic_matching_model.png)

## Results show

#### Input text
```python
test_data = [["后悔了吗","你有没有后悔"],
             ["打开自动横屏","开启移动数据"],
             ["我觉得你很聪明","你聪明我是这么觉得"]]
```
#### classification out
```
1
0
1
```
'1' indicates that the two sentences have the same meaning.
## Usage

### 1.Load data
The sample data is in /examples/bert_semantic_matching/data/

You need to define the data loading process in train.py. For example:
```python
def read_file(data_path):
    src = []
    tgt = []
    ##TODO read data file to load src and tgt, for example:
    ## src = [["article_1_1", "article_1_2"], ["article_2_1", "artile_2_2"], ......]
    ## tgt = [1, 0, ......]
    ## no matter what data you use, you need to construct the right src and tgt.
    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split("\t")
        if len(line) == 3:
            sents_tgt.append(int(line[2]))
            sents_src.append([line[0], line[1]])

    return src,tgt
```

### 2.Load model and tokenizer

```python
from flash_tran.auto_model.auto_loader import AutoLoader

# the model dir, which contains the 1.config.json, 2.pytorch_model.bin, 3.vocab.txt,
# or we will download these files from the model hub to this dir.
model_dir = "./state_dict/"
# Autoloader can build the model and tokenizer automatically.
# 'cls' is the task_name.
auto_loader = AutoLoader("cls",
                         model_dir,
                         model_name="RoBERTa-wwm-ext")
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```

### 3. Train
Then input this code in commandline to train:
```commandline
python ./train.py
```
Modify the training configuration by this code:
```python
from flagai.trainer import Trainer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(env_type="pytorch",
                  experiment_name="roberta-base-ch-semantic-matching",
                  batch_size=8, gradient_accumulation_steps=1,
                  lr = 1e-5,
                  weight_decay=1e-3,
                  epochs=10, log_interval=100, eval_interval=500,
                  load_dir=None, pytorch_device=device,
                  save_dir="checkpoints_semantic_matching",
                  save_epoch=1
                  )
```
Divide the training set validation set and create the dataset:
```python
src, tgt = read_file(data_path=train_path)
data_len = len(src)
train_size = int(data_len * 0.9)
train_src = src[: train_size]
train_tgt = tgt[: train_size]

val_src = src[train_size: ]
val_tgt = tgt[train_size: ]

train_dataset = BertClsDataset(train_src, train_tgt)
val_dataset = BertClsDataset(val_src, val_tgt)
```

### Generation
If you have already trained a model, in order to see the results more intuitively, rather than the accuracy of the validation set.
You can run the generation file.
First to modify the path of saved model.
```python
model_save_path = "./checkpoints_semantic_matching/9000/mp_rank_00_model_states.pt"
```
```commandline
python ./generate.py
```
Then you can see the generation result.

