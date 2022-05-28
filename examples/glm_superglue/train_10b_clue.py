from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments


task_name = 'tnews'
trainer = Trainer(env_type='deepspeed',
                  epochs=2,
                  batch_size=1,
                  eval_interval=1000,
                  checkpoint_activations=False,
                  fp16=True,
                  log_interval=1,
                  save_dir="./glm_superglue_en",
                  master_ip='127.0.0.1',
                  master_port=17235,
                  num_nodes=1,
                  num_gpus=2,
                  hostfile='./hostfile',
                  model_parallel_size=2,
                  deepspeed_config='./deepspeed.json',
                  training_script=__file__)

model = GLMForSingleTokenCloze.from_pretrain(download_path="/mnt/test_10b_models",
                                             model_name="glm-10b-ch")


tokenizer =  GLMLargeChTokenizer()
train_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='/mnt/datasets/yan/',
                                 dataset_type='train',
                                 tokenizer=tokenizer,
                                 cloze_eval=True)
valid_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='/mnt/datasets/yan/',
                                 dataset_type='dev',
                                 tokenizer=tokenizer,
                                 cloze_eval=True)

cl_args = CollateArguments()
cl_args.cloze_eval = True
if task_name in ['copa', 'wsc', 'record']:
    cl_args.multi_token = True

from flagai.data.dataset import ConstructSuperglueStrategy

collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name)
trainer.train(model,

              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])
