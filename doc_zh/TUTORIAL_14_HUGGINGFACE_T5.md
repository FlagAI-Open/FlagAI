# 支持 Huggingface t5


```python
import sys
from easybigmodel.trainer import Trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset
import torch

## Inheriant the Trainer
## overload the forward_step function
class MyTrainer(Trainer):

    def forward_step(self, data, model, mems):
        """
        Args:
            data: a dict contains a batch of inputs
        return:
            output: a dict contains `loss`
        """
        model_outputs = model(**data)
        output = {}
        output['loss'] = model_outputs.loss
        output['logits'] = model_outputs.logits
        output['hidden_states'] = model_outputs.decoder_hidden_states
        return output

# get a customized trainer instance
trainer = MyTrainer(
    env_type='pytorch',
    epochs=1,
    batch_size=4,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    pytorch_device='cuda:0',
    load_dir=None,
    lr=1e-4,
    fp16=False)

# using huggingface transformers to get tokenizer and models
model_name = 't5-11b'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("loading model & tokenizer is done!")
src_dir = 'train_inputs.txt'
tgt_dir = 'train_targets.txt'
model_dir = "./t5-11b"  # 模型位置
maxlen = 1024


def read_file():
    src = []
    tgt = []

    with open(src_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            src.append(line.strip('\n').lower())

    with open(tgt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n').lower())
    return src, tgt


class BertSeq2seqDataset(Dataset):

    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
        super().__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        inputs = tokenizer(src)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt)
        output = {}
        output['input_ids'] = inputs.input_ids
        output['labels'] = labels.input_ids
        return output

    def __len__(self):
        return len(self.sents_src)


def seq2seq_collate_fn(batch):

    def padding(indice, max_length, pad_idx=0):

        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice)

    token_ids = [data["input_ids"] for data in batch]
    max_length_tk = max([len(t) for t in token_ids])
    labels = [data["labels"] for data in batch]
    max_length_lb = max([len(t) for t in labels])

    token_ids_padded = padding(token_ids, max_length_tk)
    labels_padded = padding(labels, max_length_lb)

    data = {"input_ids": token_ids_padded, "labels": labels_padded}

    return data


sents_src, sents_tgt = read_file()
data_len = len(sents_tgt)
train_size = int(data_len * 0.8)
train_src = sents_src[:train_size]
train_tgt = sents_tgt[:train_size]

val_src = sents_src[train_size:]
val_tgt = sents_tgt[train_size:]

train_dataset = BertSeq2seqDataset(train_src,
                                   train_tgt,
                                   tokenizer=tokenizer,
                                   maxlen=maxlen)
val_dataset = BertSeq2seqDataset(val_src,
                                 val_tgt,
                                 tokenizer=tokenizer,
                                 maxlen=maxlen)
## Training
trainer.train(model,
              train_dataset=train_dataset,
              collate_fn=seq2seq_collate_fn)

```

## 加速训练的技巧
我们可能不会在V100 32G上运行t5-11b。所以，我们需要一些技巧来减少GPU内存的使用。
### 第一步：fp16
把模型参数变为 `fp16`
```python
trainer = MyTrainer(
    env_type='pytorch',
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    pytorch_device='cuda:0',
    load_dir=None,
    lr=1e-4,
    fp16=True) # change to `True`
```  
### 第二步：梯度重计算（checkpoint）
在forward阶段不将中间结果保存。我们可以运行`batch size`=1的t5-11b。
现在，我们可以用 `gradient_accumulation_steps` train/finetune 一个 t5-11b。
```python
model.gradient_checkpointing = True
```  
### 第三步：数据并行(DDP)
为了增加batch size，我们可以在多个GPU上使用数据并行。
```python
trainer = Trainer(
    env_type="pytorchDDP",
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    load_dir=None,
    lr=1e-4,
    fp16=True
    checkpoint_activations=False,
    # The following six options is for pytorchDDP
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    hostfile='hostfile', #  hostfile setup the number of nodes & gpus
    training_script=__file__,
)
```
### 第四步：数据并行（deepspeed）
通过使用`cpuoffload` 和`stage2`，将单个gpu上的  `batch size` 增加到 `4`。

```python
trainer = Trainer(
    env_type="deepspeed", # env_type
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    load_dir=None,
    lr=1e-4,
    fp16=True
    checkpoint_activations=False,
    # parallel settings
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    hostfile='hostfile',
    training_script=__file__,
    # deepspeed
    deepspeed_config='deepspeed.json'
)
```
### 第五步：模型并行（deepspeed + megatron-lm）

```python
trainer = Trainer(
    env_type="deepspeed", # env_type
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    load_dir=None,
    lr=1e-4,
    fp16=True
    checkpoint_activations=False,
    # parallel settings
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    hostfile='hostfile',
    training_script=__file__,
    hostfile='hostfile',
    # deepspeed
    deepspeed_config='deepspeed.json',
    # megatron-lm
    model_paralle_size = 2
)
```