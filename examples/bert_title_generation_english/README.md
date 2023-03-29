# BERT for seq2seq tasks such as title generation english

# Flash Attention(https://github.com/HazyResearch/flash-attention)
We already integrated flash attention for BERT models. This task was speeduped 10%+ using flash attention in our testing(A100, one gpu).
## How to use
We can easily use the flash attention as follows.
### 1. add "enable_flash_atten": false into model config.json file.
### 2. Dataset add "flash_atten" flags into train_flash_atten.py
### Training
```python
import sys
import os
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer
from flagai.data.collate_utils import seq2seq_collate_fn as title_generation_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cur_dir = os.path.dirname(os.path.abspath(__file__))
train_path = cur_dir + "/data/news.tsv"
# single gpu
trainer = Trainer(
    env_type="pytorch",
    experiment_name="bert-title-generation",
    batch_size=32,
    gradient_accumulation_steps=1,
    lr=1e-5,
    weight_decay=1e-3,
    epochs=10,
    log_interval=1,
    eval_interval=10,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints-bert-title-generation-en",
    checkpoint_activations=False,
    save_interval=1000,
    fp16 = True)

model_dir = "../state_dict/"  # download_path for the model 
os.makedirs(model_dir, exist_ok=True)
maxlen = 256

auto_loader = AutoLoader(
    "title-generation",
    model_name="BERT-base-en",
    model_dir=model_dir,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()


def read_file():
    src = []
    tgt = []

    index = 0
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
```
