
## 在GPU上微调模型

让我们按照如下所示的步骤来对CLUE上的AFQMC以及TNEWS数据集进行微调

```python
from easybigmodel.model.glm_model import GLMModel,GLMForSequenceClassification
from easybigmodel.data.tokenizer import GLMLargeEnWordPieceTokenizer, GLMLargeChTokenizer
from easybigmodel.metrics import *

from easybigmodel.data.dataset import SuperGlueDataset
from easybigmodel.test_utils import CollateArguments
from easybigmodel.data.dataset.superglue.control import  DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
import unittest
from easybigmodel.data.dataset import ConstructSuperglueStrategy
for task_name in ['afqmc', 'tnews']:
    trainer = Trainer(env_type='pytorch',
                      epochs=1,
                      batch_size=4,
                      eval_interval=100,
                      log_interval=50,
                      experiment_name='glm_large',
                      pytorch_device='cuda',
                      load_dir=None,
                      lr=1e-4,
                      save_epoch=10
                      )
    print("downloading...")

    cl_args = CollateArguments()
    cl_args.multi_token = task_name in MULTI_TOKEN_TASKS

    model_name='GLM-large-ch'
    tokenizer = GLMLargeChTokenizer(add_block_symbols=True, add_task_mask=False,
                                      add_decoder_mask=False, fix_command_token=True)


    if cl_args.multi_token:
        model = GLMForMultiTokenCloze.from_pretrain(model_name=model_name)
    else:
        model = GLMForSingleTokenCloze.from_pretrain(model_name=model_name)

    train_dataset = SuperGlueDataset(task_name=task_name, data_dir='./datasets/', dataset_type='train',
                                     tokenizer=tokenizer)
    train_dataset.example_list = train_dataset.example_list[:2]

    collate_fn = ConstructSuperglueStrategy(cl_args, tokenizer, task_name=task_name)

    valid_dataset = SuperGlueDataset(task_name=task_name, data_dir='/mnt/datasets/yan/', dataset_type='dev',
                                     tokenizer=tokenizer)
    valid_dataset.example_list = valid_dataset.example_list[:2]

    metric_methods = DEFAULT_METRICS[task_name]
    trainer.train(model, collate_fn=collate_fn,
                  train_dataset=train_dataset, valid_dataset=valid_dataset,
                  metric_methods=metric_methods)
```

```shell
>>> python train.py
```


## 在deepspeed上进行微调

变更到deepspeed只需要在trainer里修改一些参数即可

```python
 trainer = Trainer(env_type='deepspeed', # change env_type
                    epochs=1,
                    batch_size=4,
                    eval_interval=100,
                    log_interval=50,
                    experiment_name='glm_large',
                    load_dir=None
                  
                    # parallel settings
                    master_ip='127.0.0.1',
                    master_port=17750,
                    num_nodes=1,
                    num_gpus=2,
                    hostfile='hostfile',
                    training_script=__file__,
                    # deepspeed
                    deepspeed_config='deepspeed.json')
```
```shell
>>> python train.py
```
hostfile: 所有节点的设置ip
```shell
127.0.0.1 slots=2 # slots 代表可用的GPU数量
```
deepspeed_config: deepspeed的参数设置放在这儿
```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 50,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 1, # important
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e7,
    "allgather_bucket_size": 5e7,
    "cpu_offload": true # important
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
      "lr": 5e-6,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-8,
      "weight_decay": 1e-2
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false,
}


```
