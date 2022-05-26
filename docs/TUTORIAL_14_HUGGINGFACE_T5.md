# Support Huggingface t5


```python
import sys
from flagai.trainer import Trainer
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
    experiment_name='t5-3b',
    pytorch_device='cuda:0',
    load_dir=None,
    lr=1e-4,
    fp16=False)

# using huggingface transformers to get tokenizer and models
model_name = 't5-3b'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("loading model & tokenizer is done!")
src_dir = 'train_inputs.txt'
tgt_dir = 'train_targets.txt'
model_dir = "./t5-3b"  # 模型位置
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
        super(BertSeq2seqDataset, self).__init__()
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

## Tricks for speedup training
We may not run a t5-3b on a V100 32G. So, we need some tricks to cut down the GPU memory usage.
### step1.fp16
Model parameters turned to `fp16`
```python
trainer = MyTrainer(
    env_type='pytorch',
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
    pytorch_device='cuda:0',
    load_dir=None,
    lr=1e-4,
    fp16=True) # change to `True`
```  
### step2.gradient recomputation(checkpoints)
Do not save the itermedia results in forward stage. Now you may run t5-3b with `batch size`=1.
Now, we can train/finetune a t5-3b with `gradient_accumulation_steps`.
```python
trainer = MyTrainer(
    env_type='pytorch',
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
    pytorch_device='cuda:0',
    load_dir=None,
    lr=1e-4,
    fp16=True
    checkpoint_activations = True) # setting as `True`
```  
### step3. data parallel (DDP)
To multiply your batch size, we can use data paralle on multiple GPUs.
```python
trainer = Trainer(
    env_type="pytorchDDP",
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
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
### step4. data parallel (deepspeed)
With `cpuoffload` and `stage2`, increase the `batch size` on single gpu to `4`.
```python
trainer = Trainer(
    env_type="deepspeed", # env_type
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
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

### step5. model parallel (deepspeed + megatron-lm)
Open your imagenation.
```python
trainer = Trainer(
    env_type="deepspeed", # env_type
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
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