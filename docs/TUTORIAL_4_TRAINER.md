# Trainer
- [Trainer](#trainer)
  - [Getting Started](#getting-started)
  - [env_type in Trainer](#env_type-in-trainer)
  - [customize a Trainer](#customize-a-trainer)
  - [Single node cpu/gpu](#single-node-cpugpu)
  - [fp16](#fp16)
  - [Gradient recomputation](#gradient-recomputation)
    - [GLM-10b-ch](#glm-10b-ch)
    - [huggingface t5-11b](#huggingface-t5-11b)
  - [Training huggingface t5-11b example with Trainer](#training-huggingface-t5-11b-example-with-trainer)
  - [Parallel training](#parallel-training)
    - [deepspeed](#deepspeed)
    - [pytorchDDP](#pytorchddp)
    - [deepspeed + megatron-lm](#deepspeed--megatron-lm)
- [EnvTrainer](#envtrainer)


The Trainer class provides APIs for training with multiple parallel frameworks. The API supports distributed training with Pytorch DDP/Deepspeed on multiple GPUs, as well as mixed parallel distributed training with Megatron-LM+Deepspeed, and mixed precision via NVIDIA Apex.

## Getting Started
Trainer includes basic training loops that support the above features. Two steps to use a Trainer: initialization and excution. Refer to the code in the directory `examples/glm_superglue` .

## env_type in Trainer
There is a parameter env_type in Trainer that controls whether the training is distributed or not.

```shell
The enviroment type for training. Will default to 'pytorch'.
env_type: `pytorch`, `pytorchDDP`, `deepspeed`, `deepspeed+mpu`
            pytorch: single node cpu/gpu
            pytorchDDP: single-/multi- node gpu <data parallel>
            deepspeed: single-/multi- node gpu <data/pipline parallel>
            deepspeed+mpu: single-/multi- node gpu <data parallel + model parallel>
```                          
## customize a Trainer
When using a custom model, when the input and output of the model are inconsistent with the behavior of the model in the FlagAI framework (refer to the introduction of the [model forward function](TUTORIAL_3_MODEL.md#forward-function)), a custom Trainer is required for training. To customize Trainer to quickly support custom models, you can inherent Trainer and override the forward_step method. Note: the return of the forward_step method is a dict
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
When the model parameters are large and the GPU memory space is very tight, the memory usage can be reduced by converting the fp32 parameters to fp16. FlagAI can realize such transformer with change of a parameter.
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
 
Below, we give two examples of tens of billions of models to enable gradient recalculation, one is to load the 10-billion parameter model from FlagAI, and the other is to load the tens of billions model of huggingface.

### GLM-10b-ch
```python
#download model from modelhub and activate gradient recompuatation
from flagai.model.glm_model import GLMModel
model = GLMModel.from_pretrain(download_path="./state_dict", model_name="GLM-large-ch", checkpoint_activations=True)
```

### huggingface t5-11b
t5-11b paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-11b')
model = T5ForConditionalGeneration.from_pretrained('t5-11b')
model.gradient_checkpointing = True
```
To demonstrate the scalability of FlagAI's Trainer, let's use the training `T5-11b` model as an example. Note: The weight of models with a scale of more than 10 billion is **20G+**, and a single `gpu >= V100` is required.

## Training huggingface t5-11b example with Trainer 
FlagAI example：`examples/t5_huggingface`

```python
# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.trainer import Trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset
import torch


class MyTrainer(Trainer):

    def forward_step(self, data, model, mems):

        model_outputs = model(**data)
        output = {}
        output['loss'] = model_outputs.loss
        output['logits'] = model_outputs.logits
        output['hidden_states'] = model_outputs.decoder_hidden_states
        return output


trainer = MyTrainer(
    env_type='deepspeed',
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    load_dir=None,
    lr=1e-4
    # parameters for pytorchDDP
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=1,
    training_script=__file__,
    # deepspeed
    deepspeed_config='deepspeed.json'
)

model_name = 't5-11b'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.gradient_checkpointing = True

print("loading model & tokenizer is done!")
src_dir = './data/train.src'
tgt_dir = './data/train.tgt'
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


class T5Seq2seqDataset(Dataset):

    def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
        super(T5Seq2seqDataset, self).__init__()
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
train_src = sents_src[:train_size][:200]
train_tgt = sents_tgt[:train_size][:200]

val_src = sents_src[train_size:]
val_tgt = sents_tgt[train_size:]

train_dataset = T5Seq2seqDataset(train_src,
                                 train_tgt,
                                 tokenizer=tokenizer,
                                 maxlen=maxlen)
val_dataset = T5Seq2seqDataset(val_src,
                               val_tgt,
                               tokenizer=tokenizer,
                               maxlen=maxlen)

trainer.train(model,
              train_dataset=train_dataset,
              collate_fn=seq2seq_collate_fn)
```

## Parallel training
For speedup model training, FlagAI supports three types of paralleled training, but in the example of training T5-11b, only the the `deepspeed` framework can be used.

### deepspeed
Deepspeed provides the cpu-offload optimizer , which can greatly reduce the occupation of gpu memory.  The configuration file of `deepspeed.json` is as follows,
```shell
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e7,
    "allgather_bucket_size": 5e7,
    "cpu_offload": true 
  },
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0004,
      "weight_decay": 0.01,
      "betas": [
        0.9,
        0.98
      ],
      "eps": 1e-6
    }
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false
}
```
`deepspeed_config` can be find in  `examples/t5_huggingface/deepspeed.json`. `stage` and `cpu_offload` are two key parameters.

`hostfile` can be is ignored in single node setting.

### pytorchDDP
DistributedDataParallel (DDP) can be used when the size of model parameters <1 billion, e.g., `t5-base`. We can activate the framework by setting `env_type` = `pytorchDDP`.

```python
trainer = MyTrainer(
    env_type='pytorchDDP',
    epochs=1,
    batch_size=1,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-base',
    load_dir=None,
    lr=1e-4
    # parameters for pytorchDDP
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=1,
    hostfile='./hostfile',
    training_script=__file__,
)
```
### deepspeed + megatron-lm
Now the 10-billion model GLM-10-ch adopts the model-parallel technology of `Megatron-LM` and the data-parallel technology of `deepspeed`. When the size of model parameters is above 10-billion, it is difficult to load a model and all the intermediate variables during training in a single gpu. To this end, `Megatron-LM` provides a model-parallel method. The main idea is to segment the matrix according to rows/columns. FlagAI converts the model to the `Megatron-LM` version.
As follows, FlagAI support Megatron-LM version of models (GLM, T5, BERT [including RoBERTa], GPT2), as long as the environment variable is modified to `deepspeed+mpu` in the configuration file, the model-parallel version can be activated.
For the huggingface models, there is no `Megatron-LM` support in FlagAI.
```python
trainer = MyTrainer(
    env_type="deepspeed+mpu", # env_type
    epochs=1,
    batch_size=8,
    eval_interval=10,
    log_interval=10,
    experiment_name='GLM-10b-ch',
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

# EnvTrainer

To input the parameters easier, we provided the EnvTrainer to replace the original Tranier.

Taking the code for example:
```python
# train.py
import torch
from flagai.env_args import EnvArgs
from flagai.env_trainer import EnvTrainer

lr = 2e-5
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_args = EnvArgs(
    env_type="pytorch",
    experiment_name="vit-cifar100-single_gpu",
    batch_size=150,
    num_gpus=1,
    gradient_accumulation_steps=1,
    lr=lr,
    weight_decay=1e-5,
    epochs=n_epochs,
    log_interval=100,
    eval_interval=1000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_vit_cifar100_single_gpu",
    save_interval=1000,
    num_checkpoints=1,
)

env_args.add_arg(arg_name="test1", default=0, type=int, )
env_args_parse = env_args.parse_args()
trainer = EnvTrainer(env_args)
```

When you run the train.py file, you can modify the input parameters through command line.
```commandline
python train.py --batch_size=8 --epochs=10
```
If you need to add additional parameters, you can call the function:
```python
env_args.add_arg(arg_name="test1", default=0, type=int, )
```
Then you can run the train.py file in the following command:
```commandline
python train.py --test1=1
```

More examples in :

1. [vit-env-trainer](https://github.com/BAAI-Open/FlagAI/tree/master/examples/vit_cifar100/train_env_trainer.py)

2. [glm-title-generation-env-trainer](https://github.com/BAAI-Open/FlagAI/tree/master/examples/glm_title_generation/train_env_trainer.py)


# Run with pytorchDDP launcher or deepspeed launcher
If you use multiple GPU to train models, you can run the train.py directly which to call the launcher in FlagAI Trainer.
```commandline
python train.py
```
In addition, you also can use the pytorchDDP and deepspeed launcher to run, as example:

### pytorchDDP
```commandline
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 17750 train_env_trainer.py --not_call_launch
```
### deepspeed
```commandline
python -m deepspeed.launcher.launch  --master_addr=172.31.125.121 --master_port=17500 train.py --not_call_launch
```