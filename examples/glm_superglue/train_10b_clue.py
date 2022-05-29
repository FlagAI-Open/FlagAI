import os
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset import ConstructSuperglueStrategy


task_name = 'tnews'
trainer = Trainer(env_type='deepspeed',
                  epochs=2,
                  batch_size=4,
                  eval_interval=10,
                  checkpoint_activations=False,
                  fp16=True,
                  log_interval=1,
                  save_dir="./glm_superglue_en",
                  master_ip='127.0.0.1',
                  master_port=17237,
                  num_nodes=1,
                  num_gpus=2,
                  hostfile='./hostfile',
                  model_parallel_size=2,
                  deepspeed_config=os.path.dirname(os.path.abspath(__file__))+'/deepspeed.json',
                  training_script=__file__)

model = GLMForSingleTokenCloze.from_pretrain(download_path="/mnt/test_10b_models",
                                             model_name="glm-10b-ch")


tokenizer =  GLMLargeChTokenizer()

train_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='./datasets/',
                                 dataset_type='train',
                                 tokenizer=tokenizer,
                                 cloze_eval=True)
valid_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='./datasets/',
                                 dataset_type='dev',
                                 tokenizer=tokenizer,
                                 cloze_eval=True)

cl_args = CollateArguments()
cl_args.cloze_eval = True
cl_args.multi_token = False

collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name)
trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])