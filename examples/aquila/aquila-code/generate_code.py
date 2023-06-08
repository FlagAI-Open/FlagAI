import torch
import os
import argparse
import sys
sys.path.append("/data2/yzd/workspace/FlagAI")
from flagai import mpu
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np
from flagai.model.predictor.predictor import Predictor
from pathlib import Path 
from flagai.data.tokenizer import Tokenizer
import time
import torch.distributed as dist
import json 
import json, datetime

import os

model_dir = "./checkpoints_in"
device = "cuda"

print(f"building model...")
loader = AutoLoader("lm", model_name="aquilacode-7b-nv",#,"llama-7b-en", 
                    only_download_config=True, 
                    use_cache=True, 
                    fp16=True,
                    model_dir=model_dir)

model = loader.get_model()
tokenizer = loader.get_tokenizer()

# import pdb;pdb.set_trace()
# ckpt = torch.load('./checkpoints_in/aquilacode-7b-nv/pytorch_model.bin', map_location=torch.device('cpu'))
# # print(ckpt)
# model.load_state_dict(ckpt, strict=True)

model.eval()

model.to(device)

vocab = tokenizer.get_vocab()

id2word = {v:k for k, v in vocab.items()}

predictor = Predictor(model, tokenizer)

max_new_tokens = 256

test_file = "./datasets/code_test.txt"
with open(test_file) as fin:
    prompt = '\n'+fin.read()+'\n'

input_ids = tokenizer.encode_plus_non_glm(prompt)["input_ids"][:-1]
input_length = len(input_ids)

max_length = input_length+max_new_tokens
with torch.no_grad():

    # prompt = "#用户#" + prompt + " " + "#ai助手#"

    prompt = '''"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."''' + '''### Human: ''' + prompt.strip() + '''### Assistant:'''        
    res = predictor.predict_generate_randomsample(prompt, 
                                                    out_max_length=max_length, 
                                                    top_p=0.95, 
                                                    temperature=t0.7)
    print(res)