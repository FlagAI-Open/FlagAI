from flagai.data.dataset import ConstructSuperglueStrategy, SuperGlueDataset
from flagai.data.dataset.superglue.control import (CH_TASKS, DEFAULT_METRICS,
                                                   MULTI_TOKEN_TASKS)
from flagai.data.tokenizer import Tokenizer
from flagai.model.glm_model import GLMForSequenceClassification
from flagai.test_utils import CollateArguments
from flagai.trainer import Trainer

task_name = "tnews"
model_name = 'GLM-large-ch'
cl_args = CollateArguments()
cl_args.cloze_eval = False
cl_args.multi_token = task_name in MULTI_TOKEN_TASKS


tokenizer = Tokenizer.from_pretrained(model_name)

model = GLMForSequenceClassification.from_pretrain(model_name=model_name, spell_length=2,
                                                    class_num=3, tune_prefix_layers=1)



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
