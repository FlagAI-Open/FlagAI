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

cl_args = CollateArguments()
cl_args.multi_token = task_name in MULTI_TOKEN_TASKS
if task_name in CH_TASKS:
    model_name = 'GLM-large-ch'
    add_block_symbols=True,
else:
    model_name = 'GLM-large-en'
tokenizer = Tokenizer.from_pretrained(model_name)

model = GLMForSingleTokenCloze.from_pretrain(download_path="./checkpoints",
                                             model_name=model_name)
# model_save_path = "/home/yanzhaodong/anhforth/test/FlagAI/examples/glm_superglue/checkpoints/20000_save/pytorch_model.bin"
# model.load_state_dict(torch.load(model_save_path, map_location="cuda")["module"])  
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

# Deepspeed parallel trainer
# trainer = Trainer(env_type='deepspeed',
#                   epochs=10000000,
#                   batch_size=16,
#                   gradient_accumulation_steps=5,
#                   checkpoint_activations=True,
#                   eval_interval=False,
#                   log_interval=100,
#                   fp16=True,
#                   save_interval=10000,
#                   experiment_name='glm_large',
#                   load_dir=None,
#                   num_nodes=1,
#                   num_gpus=2,
#                   hostfile='./hostfile',
#                   deepspeed_config='./deepspeed.json',
#                   lr=1e-4,
#                   training_script=__file__)
# Single-GPU trainer
trainer = Trainer(env_type='pytorch',
                    epochs=100,
                    batch_size=1,
                    eval_interval=100,
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
