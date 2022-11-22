# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import argparse
import os
import random

import numpy as np
import torch

from flagai import mpu
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

# run script : python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 glm_blank_filling_QA_ch_mutigpu.py
os.environ["ENV_TYPE"] = "deepspeed+mpu"
model_parallel_size = 2
world_size = 2
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

loader = AutoLoader("lm", model_name="GLM-large-ch")
model = loader.get_model()
model.half()
tokenizer = loader.get_tokenizer()
model.parallel_output = False
model.eval()
model.to(device)

torch.distributed.barrier(group=mpu.get_model_parallel_group())

text = '问题：啤酒伤胃吗？回答：[gMASK]'

predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text)
if mpu.get_model_parallel_rank() == 0:
    print(f"pred is {out}")
