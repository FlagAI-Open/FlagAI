# Trainer

The Trainer class provides APIs for training with multiple parallel frameworks. The API supports distributed training with Pytorch DDP/Deepspeed on multiple GPUs, as well as mixed parallel distributed training with Megatron-LM+Deepspeed, and mixed precision via NVIDIA Apex.

## Getting Started
Trainer includes basic training loops that support the above features. To customize the behavior, you can subclass them and override the forward_step method:
The return of the forward_step method is a dict

```python
from flagai.trainer import Trainer

class MyTrainer(Trainer):

    def forward_step(self, data, model, mems):

        model_outputs = model(**data)
        output = {}
        output['loss'] = model_outputs.loss
        output['logits'] = model_outputs.logits
        output['hidden_states'] = model_outputs.decoder_hidden_states
        return output
```

There is a parameter env_type in Trainer that controls whether the training is distributed or not.

```shell
The enviroment type for training. Will default to 'pytorch'.
env_type: `pytorch`, `pytorchDDP`, `deepspeed`, `deepspeed+mpu`
            pytorch: single node cpu/gpu
            pytorchDDP: single-/multi- node gpu <data parallel>
            deepspeed: single-/multi- node gpu <data/pipline parallel>
            deepspeed+mpu: single-/multi- node gpu <data parallel + model parallel>
```                          

## Single node cpu/gpu

```python
trainer = MyTrainer(
    env_type='pytorch',
    epochs=1,
    batch_size=4,
    eval_interval=100000,
    log_interval=10,
    experiment_name='t5-11b',
    pytorch_device='cpu',
    load_dir=None,
    lr=1e-4)
```
Specify the graphics card/cpu settings as pytorch_device, 'cpu', 'cuda:0', etc.
## fp16

```python
Model parameters turned to fp16
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
## Gradient recomputation
Do not save the Intermediate results in the forward stage. Paper: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174v2)
 
Now, we can train/finetune a t5-11b with gradient_accumulation_steps. 

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-11b')
model = T5ForConditionalGeneration.from_pretrained('t5-11b')
model.gradient_checkpointing = True
```
## Support huggingface model
The example directory location of the FlagAI project：examples/t5_huggingface
t5-11b paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)

The example train t5-11b on a toy dataset.  You need  a device > 30G memory to run the example. If your have multiple devices,  please check out the following session for speedup.

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

## Distributed training
To accelerate your training/finetuning process, FlagAI integrate three popular parallel frameworks for deep neural networks.
### pytorchDDP
DistributedDataParallel (DDP) link
We can use distributed training by modifying a few parameters. The launch call is directly encapsulated inside, and python script_name.py [script_name: is the name of the training script] can be run directly.

```python
trainer = MyTrainer(
    env_type='pytorchDDP',
    epochs=1,
    batch_size=4,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    load_dir=None,
    lr=1e-4
    # parameters for pytorchDDP
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    hostfile='./hostfile',
    training_script=__file__,
)
```
### deepspeed
Compared with pytorch, deepspeed mainly optimizes the optimizer.
It provides an optimizer of cpu-offload, which can greatly reduce the occupation of gpu video memory.
For the relevant settings of the deepspeed optimizer, the configuration file of deepspeed.json needs to be provided

```python
trainer = MyTrainer(
    env_type='pytorchDDP',
    epochs=1,
    batch_size=4,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    load_dir=None,
    lr=1e-4
    # parameters for pytorchDDP
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    hostfile='./hostfile',
    training_script=__file__,
    # deepspeed
    deepspeed_config='deepspeed.json'
)
```

Deepspeed config file reference [examples/t5_huggingface/deepspeed.json]
Among them, the main parameters are stage and cpu offload

### deepspeed + megatron-lm

Now the tens of billions of model GLM adopts the model parallel technology of Megatron-LM. When the model parameter scaling is above 10b, it is difficult to load a single model and all the intermediate variables during training in the video memory of a single card. To this end, Megatron-LM provides a Tensor segmentation method. The main idea is to segment the matrix according to rows/columns. FlagAI link, convert the model to the Megatron-LM version.
As follows, FlagAI's internal models (GLM, T5, BERT [including RoBERTa], GPT2) support Megatron-LM. As long as the environment variable is changed to deepspeed+mpu in the configuration file, the model parallel function can be started.
For the huggingface version of the model, there is no support for model parallelism.

```python
trainer = MyTrainer(
    env_type="deepspeed+mpu", # env_type
    epochs=1,
    batch_size=8,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    load_dir=None,
    lr=1e-4,
    # parallel settings
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=4,
    hostfile='hostfile',
    training_script=__file__,
    # deepspeed
    deepspeed_config='deepspeed.json',
    # megatron-lm
    model_paralle_size = 2
)
```
