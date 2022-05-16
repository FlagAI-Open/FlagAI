import torch.utils.data
from torch.optim import Adam
from flagai.schedulers import AnnealingLR
from flagai.trainer import Trainer
from flagai.model.glm_model import GLMModel, GLMForSingleTokenCloze
from flagai.data.tokenizer.glm_large_en.glm_large_en_tokenizer import GLMLargeEnWordPieceTokenizer
from flagai.metrics import accuracy_metric
from flagai.data.dataset import SuperGlueDataset
from flagai.test_utils import CollateArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(env_type='pytorch',
                  pytorch_device=device,
                  epochs=2,
                  batch_size=8,
                  eval_interval=1000,
                  log_interval=500,
                  save_dir="./glm_superglue_en")

model = GLMForSingleTokenCloze.from_pretrain(download_path="./state_dict",
                                             model_name="glm_large_en")

optimizer = Adam(model.parameters())

tokenizer = GLMLargeEnWordPieceTokenizer()
task_name = 'boolq'
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
lr_scheduler = AnnealingLR(optimizer,
                           start_lr=1e-5,
                           warmup_iter=int(0.1 * 2 * len(train_dataset)),
                           decay_style='linear',
                           num_iters=2 * len(train_dataset))
trainer.train(model,
              optimizer=optimizer,
              lr_scheduler=lr_scheduler,
              train_dataset=train_dataset,
              valid_dataset=valid_dataset,
              collate_fn=collate_fn,
              metric_methods=[["acc", accuracy_metric]])
