# 为模型和数据并行训练定制训练器
- [Trainer](#trainer)
  - [入门](#入门)
  - [Trainer主要参数env_type](#trainer主要参数env_type)
  - [自定义Trainer](#自定义trainer)
  - [单节点cpu/gpu](#单节点cpugpu)
  - [fp16](#fp16)
  - [梯度重计算](#梯度重计算)
    - [GLM-10b-ch](#glm-10b-ch)
    - [huggingface t5-11b](#huggingface-t5-11b)
  - [完整的用Trainer训练huggingface t5-11b例子](#完整的用trainer训练huggingface-t5-11b例子)
  - [分布式训练](#分布式训练)
    - [deepspeed](#deepspeed)
    - [pytorchDDP](#pytorchddp)
    - [deepspeed + megatron-lm](#deepspeed--megatron-lm)
- [EnvTrainer](#EnvTrainer)

Trainer 类提供了API用于多种并行框架的训练。API 支持在多个 GPU上使用Pytorch DDP/Deepspeed进行分布式训练，同时支持Megatron-LM+Deepspeed的混合并行分布式训练，同时也通过 NVIDIA Apex 实现混合精度。
## 入门
Trainer 包含支持上述功能的基本训练循环。Trainer使用分成两个步骤，初始化和调用. 参考 `examples/glm_superglue` 目录下的代码.

## Trainer主要参数env_type
Trainer 中有一个控制是否分布式训练的参数  env_type，
```shell
The enviroment type for training. Will default to 'pytorch'.
env_type: `pytorch`, `pytorchDDP`, `deepspeed`, `deepspeed+mpu`
            pytorch: single node cpu/gpu
            pytorchDDP: single-/multi- node gpu <data parallel>
            deepspeed: single-/multi- node gpu <data/pipline parallel>
            deepspeed+mpu: single-/multi- node gpu <data parallel + model parallel>
```            

## 自定义Trainer
在使用自定义模型的时候，模型的输入输出与`FlagAI`框架中模型的行为不一致时（参考 [模型forward函数介绍](TUTORIAL_3_MODEL.md#模型的forward-函数)），需要自定义Trainer来进行训练。要自定义`Trainer`来快速支持自定义模型，您可以它们进行子类化并覆盖`forward_step`方法. 这里需要注意：`forward_step`方法的返回是一个dict
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

## 单节点cpu/gpu
可以看到这里的Trainer使用的是`pytorch` `cpu`的环境
```python
trainer = MyTrainer(
    env_type='pytorch',
    epochs=1,
    batch_size=4,
    eval_interval=10,
    log_interval=10,
    experiment_name='t5-11b',
    pytorch_device='cpu',
    load_dir=None,
    lr=1e-4)
```
指定显卡/cpu的设置为pytorch_device, 'cpu', 'cuda:0'等
## fp16 
在模型参数大，而显存空间又很紧张的时候，可以通过将`fp32`的参数转化为`fp16`的参数来降低显存的使用量.FlagAI可以通过一个参数来启动。
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
## 梯度重计算
在前向过程中不保存中间变量，节约GPU显存.  Paper: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174v2)。下面，我们举了两个百亿模型开启梯度重计算的例子，一个是从FlagAI加载百亿参数模型，一个是加载huggingface的百亿模型。

### GLM-10b-ch
```python
#从modelhub下载GLM-10b-ch 模型并开启梯度重计算
from flagai.model.glm_model import GLMModel
model = GLMModel.from_pretrain(download_path="./state_dict", model_name="GLM-large-ch", checkpoint_activations=True)
```
### huggingface t5-11b
t5-11b模型的 paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-11b')
model = T5ForConditionalGeneration.from_pretrained('t5-11b')
model.gradient_checkpointing = True
```

为了展示FlagAI的Trainer的扩展能力，下面我们用训练T5-11b模型作为例子。**注意**: 百亿规模以上的模型权重在20G+，单块显卡需要V100 及以上的硬件。

## 完整的用Trainer训练huggingface t5-11b例子
FlagAI 项目的example目录位置：`examples/t5_huggingface`

```python
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

## 分布式训练
为了加速模型训练，飞智支持三种模式的并行方式，但是在上述训练T5-11b的例子中，只能使用`deepspeed`框架的数据并行模式。

### deepspeed
deepspeed主要在优化器方面做了优化，其提供了cpu-offload的optimizer，可以极大的降低gpu显存的占用。对deepspeed优化器的相关设置，需要提供`deepspeed.json`的配置文件 
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
`deepspeed_config`文件参考 `examples/t5_huggingface/deepspeed.json`.其中，主要的参数为 `stage` 和 `cpu_offload`.

`hostfile`在单节点时可以省略
### pytorchDDP
DistributedDataParallel (DDP) 在模型的参数规模<1 billion,比如`t5-base`的时候可以使用。我们可以通过修改`env_type`参数就能完成分布式框架的切换.

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

现在百亿级模型GLM-10-ch采用了`Megatron-LM`的模型并行技术加上`deepspeed`的数据并行技术。在模型参数scaling到10b以上级别，单卡的显存就很难将单个模型以及训练时的中间变量全部加载进来。为此，Megatron-LM提供了Tensor的切分方法，主要思想是将矩阵按照行/列进行切分。 FlagAI 将model转化为Megatron-LM版本。

如下，飞智内部模型（GLM，T5，BERT【包括RoBERTa】，GPT2）支持了Megatron-LM，只要在配置文件中将环境变量修改为deepspeed+mpu，就能启动模型并行的功能。
对于huggingface版本的模型，暂时没有提供模型并行的支持。
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

为了更容易的输入参数，我们提供了EnvTrainer代替原来的Trainer
例如：
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

运行train.py文件时，可以通过命令行修改输入参数。
```commandline
python train.py --batch_size=8 --epochs=10
```
如果你需要添加额外的参数，你可以调用这个函数:
```python
env_args.add_arg(arg_name="test1", default=0, type=int, )
```
然后你可以运行如下命令中的train.py文件:
```commandline
python train.py --test1=1
```
更多的例子可以查看 :

1. [vit-env-trainer](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/vit_cifar100/train_env_trainer.py)

2. [glm-title-generation-env-trainer](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/glm_title_generation/train_env_trainer.py)

# 使用 pytorchDDP launcher 或 deepspeed launcher 运行
如果你使用多个GPU来训练模型，你可以直接运行train.py来调用FlagAI训练器中的启动器。
```commandline
python train.py
```
另外，你也可以使用pytorchDDP和deepspeed启动器来运行，例如:
### pytorchDDP
```commandline
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 17750 train_env_trainer.py --not_call_launch
```
### deepspeed
```commandline
python -m deepspeed.launcher.launch  --master_addr=172.31.125.121 --master_port=17500 train.py --not_call_launch
```
