import torch
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSequenceClassification
from flagai.data.tokenizer import Tokenizer

from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
from flagai.data.dataset import ConstructSuperglueStrategy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = "tnews"
model_name = 'GLM-large-ch'
cl_args = CollateArguments()
cl_args.cloze_eval = False
cl_args.multi_token = task_name in MULTI_TOKEN_TASKS

tokenizer = Tokenizer.from_pretrained(model_name)

class_num = 15
model = GLMForSequenceClassification.from_pretrain(model_name=model_name, spell_length=2,
                                                   class_num=class_num, tune_prefix_layers=1)

train_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='./datasets/',
                                 dataset_type='train',
                                 tokenizer=tokenizer)

collate_fn = ConstructSuperglueStrategy(cl_args,
                                        tokenizer,
                                        task_name=task_name)

valid_dataset = SuperGlueDataset(task_name=task_name,
                                 data_dir='./datasets/',
                                 dataset_type='dev',
                                 tokenizer=tokenizer)

trainer = Trainer(env_type='pytorch',
                  pytorch_device=device,
                  epochs=2,
                  batch_size=1,
                  eval_interval=1000,
                  checkpoint_activations=False,
                  fp16=True,
                  log_interval=1,
                  save_dir="./glm_large_clue")

trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])

