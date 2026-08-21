---
license: [mit](https://model.baai.ac.cn/use-agreement)
---

# OPT

## Overview
OPT is a language model built by META AI，which is the decoder of the transformer. META AI has open sourced OPT models of different sizes.
More information at [https://github.com/facebookresearch/metaseq](https://github.com/facebookresearch/metaseq)

[OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)

| Name | Params | layers | hidden size| FFN hidden size| heads |head size|encoder|decoder| 备注|
|  -----  | ----  | -----  | ----  | -----  | ----  | -----  | -----  | -----  | -----  |
| [opt-1.3b-en](https://model.baai.ac.cn/model-detail/100042)  | 1.3B |    24|2048 | 8192|32 |64|No|Yes|
| [opt-125m-en](https://model.baai.ac.cn/model-detail/100043)  | 125M |    12|768 | 3072|12 |64|No|Yes|
| [opt-6.7b-en](https://model.baai.ac.cn/model-detail/100046)   | 6.7B | 32|4096 | 16384|32 |128| No|Yes|
| [opt-2.7b-en](https://model.baai.ac.cn/model-detail/100044)   | 2.7B | 32|2560 | 10240|32 |80| No|Yes|
| [opt-350m-en](https://model.baai.ac.cn/model-detail/100045)   | 350M | 24|1024 | 4096|16 |64| No|Yes|
| [opt-13b-en](https://model.baai.ac.cn/model-detail/100045)   | 13b | 40|5120 | 20480|40 |128| No|Yes|

Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models.


## Training data
The pre-training corpus contains a concatenation of datasets used in RoBERTa (Liu et al., 2019b), the Pile (Gao et al., 2021a), and PushShift.io Reddit (Baumgartner et al., 2020; Roller et al., 2021).
All corpora were previously collected or filtered to contain predominantly English text, but a small amount of non-English data is still present within the corpus via CommonCrawl.

## How to use

### Quick start

You can load the model to continue the text.
```python
from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader

loader = AutoLoader(task_name="lm",
                    model_name="opt-125m-en")

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()

text = "The trophy doesn’t fit in the suitcase because I think"
predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text,
                                              input_max_length=100,
                                              out_max_length=300,
                                              top_k=30,
                                              top_p=0.9,
                                              repetition_penalty=3.0)

print(f"input is {text} \n out is {out}")
```

# Multi-GPU inference
## OPT-30b

To inference by multi-GPU and model parallel, we use torch-DDP and Megatron-LM library.
### Basic step
1. Set up the parameters of model parallel, such as ```model_parallel_size``` and ```world_size```
2. Initialize torch-DDP
3. Initialize Megatron-LM, model parallel
4. Set up random seed
5. Initialize the model and tokenizer
6. Prediction
### code
```python
import torch
import os
import argparse
from flagai import mpu
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np
from flagai.model.predictor.predictor import Predictor

# run script : python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 opt_30b_en_mutigpu.py
os.environ["ENV_TYPE"] = "deepspeed+mpu"
model_parallel_size = 4
world_size = 4

os.environ["MODEL_PARALLEL_SIZE"] = str(model_parallel_size)
os.environ["WORLD_SIZE"] = str(world_size)

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',
                    type=int,
                    default=0,
                    help="local_rank")

ds_args = parser.parse_args()
local_rank = ds_args.local_rank

master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
master_port = os.environ.get('MASTER_PORT', '17501')

device = torch.device("cuda", local_rank)

def initialize_distributed():
    """Initialize torch.distributed."""
    torch.backends.cudnn.enabled = False
    # Manually set the device ids.
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'

    init_method += master_addr + ':' + master_port
    torch.distributed.init_process_group(
        backend='nccl',  # gloo
        world_size=world_size,
        rank=local_rank,
        init_method=init_method)
    mpu.initialize_model_parallel(model_parallel_size)

initialize_distributed()

set_random_seed(123)

print(f"building model...")
loader = AutoLoader("lm", model_name="opt-30b-en")
model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.half()

model.parallel_output = False
model.eval()
model.to(device)

torch.distributed.barrier(group=mpu.get_model_parallel_group())

text = """I think The Old Man and the Sea is a very good book, what do you think? I think """

predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text)
if mpu.get_model_parallel_rank() == 0:
    print(f"pred is {out}")
```
### Run script is 
```commandline
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 opt_30b_en_mutigpu.py
```
