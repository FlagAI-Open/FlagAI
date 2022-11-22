# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import os
import time

import torch

os.environ["ENV_TYPE"] = "deepspeed+mpu"
os.environ["MODEL_PARALLEL_SIZE"] = '8'
os.environ["WORLD_SIZE"] = '8'
import argparse
import random

import numpy as np

from flagai import mpu
from flagai.data.tokenizer import OPTTokenizer
from flagai.model.opt_model import OPTModel
from flagai.model.predictor.predictor import Predictor


def get_current_rank():
    with open('current_rank','r',encoding='utf8') as infile:
        line = infile.readline().strip()
    return int(line)
def set_current_rank(rank):
    with open('current_rank','w',encoding='utf8') as outfile:
        outfile.write(str(rank))

def get_current_pool():
    with open('current_pool','r',encoding='utf8') as infile:
        line = infile.readline().strip()
    return int(line)

def set_current_pool(rank):
    with open('current_pool','w',encoding='utf8') as outfile:
        outfile.write(str(rank))

# run script : python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 opt_66b_en_mutigpu.py
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',
                    type=int,
                    default=0,
                    help="local_rank")

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

ds_args = parser.parse_args()
local_rank = ds_args.local_rank

master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
master_port = os.environ.get('MASTER_PORT', '17501')

device = torch.device("cuda", local_rank)
model_parallel_size = 8
world_size = 8

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

set_current_pool(4)
set_current_rank(0)
set_random_seed(123)
torch.distributed.barrier(group=mpu.get_model_parallel_group())
tokenizer = OPTTokenizer()

while get_current_rank() != local_rank:
    time.sleep(10)
while get_current_pool() == 0:
    time.sleep(10)
set_current_pool(get_current_pool()-1)
print("loading rank {}".format(local_rank))
set_current_rank(local_rank + 1)

model = OPTModel.init_from_json('/mnt/models_xingzhaohu/opt-66b-en/config.json')
checkpoint_path = '/mnt/models_xingzhaohu/opt-66b-en/pytorch_model_{:02d}.bin'.format(local_rank)
model.half()
model.eval()
model.to(device)
model.load_weights(checkpoint_path)

print("loading rank {} finished".format(local_rank))
set_current_pool(get_current_pool()+1)
print('current rank setting is {}'.format(get_current_pool()))

torch.distributed.barrier(group=mpu.get_model_parallel_group())
text = """I think The Old Man and the Sea is a very good book, what do you think? I think """

predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text)
if mpu.get_model_parallel_rank() == 0:
    print(f"pred is {out}")
