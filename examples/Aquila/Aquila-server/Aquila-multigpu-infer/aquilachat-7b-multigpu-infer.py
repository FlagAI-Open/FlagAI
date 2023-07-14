import torch
import os
import argparse
from flagai import mpu
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np
from flagai.model.predictor.aquila import aquila_generate

os.environ["ENV_TYPE"] = "deepspeed+mpu"
## set the number of gpus, need to be consistent with the nproc_per_node.
model_parallel_size = 4
world_size = model_parallel_size

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
master_port = os.environ.get('MASTER_PORT', '17502')

device = torch.device("cuda", local_rank)

def initialize_distributed():
    """Initialize torch.distributed."""
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
loader = AutoLoader("lm", 
                    model_name="aquilachat-7b",
                    use_cache=True, 
                    fp16=True)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()
model.to(device)
torch.distributed.barrier(group=mpu.get_model_parallel_group())

text = """汽车EDR是什么？"""

from cyg_conversation import default_conversation

conv = default_conversation.copy()
conv.append_message(conv.roles[0], text)
conv.append_message(conv.roles[1], None)

tokens = tokenizer.encode_plus(f"{conv.get_prompt()}",
                                None,
                                max_length=None)['input_ids']
## TODO for few-shot inference using plain text as inputs will get better results.
## tokens = tokenizer.encode_plus(f"{text}", None, max_length=None)['input_ids']
tokens = tokens[1:-1]

with torch.no_grad():
    out = aquila_generate(tokenizer,
                            model, [text],
                            max_gen_len := 200,
                            top_p=0.95,
                            prompts_tokens=[tokens])
    if mpu.get_model_parallel_rank() == 0:

        print(f"pred is {out}")
    

