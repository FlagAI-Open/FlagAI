# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
from flagai.data.tokenizer import Tokenizer

model_dir = "./checkpoints_in"
device = "cuda"

print(f"building model...")
loader = AutoLoader("lm", model_name="aquilacode-7b-nv",
                    only_download_config=True, 
                    use_cache=True, 
                    fp16=True,
                    model_dir=model_dir)

model = loader.get_model()
tokenizer = loader.get_tokenizer()

model.eval()

model.to(device)

vocab = tokenizer.get_vocab()

id2word = {v:k for k, v in vocab.items()}

predictor = Predictor(model, tokenizer)

max_new_tokens = 256

# test_file = "./datasets/code_test.txt"
# with open(test_file) as fin:
#     prompt = '\n'+fin.read()+'\n'

texts = ["#补全代码\ndef quick_sort(x):", 
           '"""\n向用户询问他们的名字并说“你好”\m"""',
           '"""\nAsk the user for their name and say "Hello\n""""' ]

for text in texts:
    input_ids = tokenizer.encode_plus_non_glm(text)["input_ids"][:-1]
    input_length = len(input_ids)

    max_length = input_length+max_new_tokens
    with torch.no_grad():
        res = predictor.predict_generate_randomsample(prompt, 
                                                        out_max_length=max_length, 
                                                        top_p=0.95, 
                                                        temperature=0.7)
        print(res)