# 为模型和数据并行训练定制训练器

Trainer 类提供了API用于多种并行框架的训练。API 支持在多个 GPU上使用Pytorch DDP/Deepspeed进行分布式训练，同时支持Megatron-LM+Deepspeed的混合并行分布式训练，同时也通过 NVIDIA Apex 实现混合精度。
## 入门
Trainer 包含支持上述功能的基本训练循环。 要自定义行为，您可以对它们进行子类化并覆盖forward_step方法：
forward_step方法的返回是一个dict
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

Trainer 中有一个控制是否分布式训练的参数  env_type，
```shell
The enviroment type for training. Will default to 'pytorch'.
env_type: `pytorch`, `pytorchDDP`, `deepspeed`, `deepspeed+mpu`
            pytorch: single node cpu/gpu
            pytorchDDP: single-/multi- node gpu <data parallel>
            deepspeed: single-/multi- node gpu <data/pipline parallel>
            deepspeed+mpu: single-/multi- node gpu <data parallel + model parallel>
```                          

## 单节点cpu/gpu
```python
trainer = MyTrainer(
    env_type='pytorch',
    epochs=1,
    batch_size=4,
    eval_interval=100000,
    log_interval=10,
    experiment_name='t5-3b',
    pytorch_device='cpu',
    load_dir=None,
    lr=1e-4)
```
指定显卡/cpu的设置为pytorch_device, 'cpu', 'cuda:0'等
## fp16 
```python
Model parameters turned to fp16
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
## gradient recomputation(checkpoints)
Do not save the Intermediate results in the forward stage. Now you may run t5-3b with batch size=1. Paper: Training Deep Nets with Sublinear Memory Cost【https://arxiv.org/abs/1604.06174v2】
Now, we can train/finetune a t5-3b with gradient_accumulation_steps. 
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
## 支持huggingface model
FlagAI 项目的example目录位置：examples/t5_huggingface
T5-3b paper Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer[https://arxiv.org/pdf/1910.10683.pdf]
The example train T5-3b on a toy dataset.  You need  a device > 30G memory to run the example. If your have multiple devices,  please check out the following session for speedup.
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

## 分布式训练
To accelerate your training/finetuning process, FlagAI integrate three popular parallel frameworks for deep neural networks.
### pytorchDDP
DistributedDataParallel (DDP) link
我们可以通过修改几个参数就能使用分布式的训练，内部直接封装了launch的调用，直接python script_name.py  [script_name:是训练脚本的名字] 就可以直接运行。
```python
trainer = MyTrainer(
    env_type='pytorchDDP',
    epochs=1,
    batch_size=4,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
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
deepspeed相对于pytorch主要在优化器方面做了优化。
其提供了cpu-offload的optimizer，可以极大的降低gpu显存的占用。
对deepspeed优化器的相关设置，需要提供deepspeed.json的配置文件 
```python
trainer = MyTrainer(
    env_type='pytorchDDP',
    epochs=1,
    batch_size=4,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
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
Deepspeed config文件参考 [examples/t5_huggingface/deepspeed.json]
其中，主要的参数为 stage 和 cpu offload
### deepspeed + megatron-lm
现在百亿级模型GLM采用了Megatron-LM的模型并行技术。在模型参数scaling到10b以上级别，单卡的显存就很难将单个模型以及训练时的中间变量全部加载进来。为此，Megatron-LM提供了Tensor的切分方法，主要思想是将矩阵按照行/列进行切分。 FlagAI link，将model转化为Megatron-LM版本。
如下，FlagAI内部模型（GLM，T5，BERT【包括RoBERTa】，GPT2）支持了Megatron-LM，只要在配置文件中将环境变量修改为deepspeed+mpu，就能启动模型并行的功能。
对于huggingface版本的模型，暂时没有提供模型并行的支持。
```python
trainer = MyTrainer(
    env_type="deepspeed+mpu", # env_type
    epochs=1,
    batch_size=8,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-3b',
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

