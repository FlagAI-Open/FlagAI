import torch
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMForSequenceClassification
from flagai.model.glm_model import GLMForSingleTokenCloze
from flagai.data.tokenizer import Tokenizer

from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments
from flagai.data.dataset.superglue.control import DEFAULT_METRICS, MULTI_TOKEN_TASKS, CH_TASKS
from flagai.data.dataset import ConstructSuperglueStrategy


task_name = "qqp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("downloading...")

cl_args = CollateArguments()
if task_name in CH_TASKS:
    model_name = 'GLM-large-ch'
    add_block_symbols=True,
else:
    model_name = 'GLM-large-en'
tokenizer = Tokenizer.from_pretrained(model_name)

model = GLMForSingleTokenCloze.from_pretrain(download_path="./checkpoints",
                                             model_name=model_name)
                                             
# Load                                              
model_save_path = "./checkpoints/90000/pytorch_model.bin"
model.load_state_dict(
    torch.load(model_save_path, map_location=device)["module"])
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

metric_methods = DEFAULT_METRICS[task_name]

trainer = Trainer(env_type='pytorch',
                  epochs=0,
                  batch_size=4,
                  eval_interval=1,
                  log_interval=50,
                  experiment_name='glm_large',
                  pytorch_device='cuda',
                  load_dir=None,
                  lr=1e-4)

trainer.train(model,
                collate_fn=collate_fn,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                metric_methods=metric_methods)
