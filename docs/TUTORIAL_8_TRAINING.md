
## Finetuning model with GPU

Let's run finetuning over SuperGlue Dataset, as following `train.py`:

```python
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMModel,GLMForSequenceClassification
from flagai.data.tokenizer import GLMBertWordPieceTokenizer 
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments

 trainer = Trainer(env_type='pytorch',
                          epochs=1,
                          batch_size=4,
                          eval_interval=100,
                          log_interval=50,
                          experiment_name='glm_large',
                          pytorch_device='cuda:0',
                          load_dir=None
                          )
lm_model = GLMModel.from_pretrain()
model = GLMForSequenceClassification(lm_model,hidden_size=1024,hidden_dropout=False,pool_token='cls',num_class=2)
tokenizer = GLMBertWordPieceTokenizer(tokenizer_model_type='bert-large-uncased')

from flagai.data.dataset import ConstructSuperglueStrategy
cl_args = CollateArguments()
cl_args.cloze_eval=False
task_name = "boolq"
if task_name in ['copa', 'wsc', 'record', 'wanke']:
    cl_args.multi_token = True
collate_fn = ConstructSuperglueStrategy(cl_args, tokenizer, task_name=task_name)

train_dataset = SuperGlueDataset(task_name="boolq", data_dir='/mnt/datasets/yan/', dataset_type='train',
                                    tokenizer=tokenizer)
valid_dataset = SuperGlueDataset(task_name="boolq", data_dir='/mnt/datasets/yan/', dataset_type='dev',
                                    tokenizer=tokenizer)


trainer.train(model, collate_fn=collate_fn,
                train_dataset=train_dataset, valid_dataset=valid_dataset, 
                eval_metrics=accuracy_metric)
```

```shell
>>> python train.py
```

Done! The  results may be around 80%

## Finetunine with deepspeed

we only change some settings in Trainer & run 

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
hostfile: setting ip for all nodes
```shell
127.0.0.1 slots=2 # slots is the number of your gpus
```
deepspeed_config: arguments for deepspeed settings
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
