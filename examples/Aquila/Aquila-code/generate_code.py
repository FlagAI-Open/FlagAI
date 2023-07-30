# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import os
from flagai import mpu
import sys
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np

from flagai.model.predictor.predictor import Predictor
from flagai.data.tokenizer import Tokenizer

model_dir = "./checkpoints_in"
device = "cuda"

print(f"building model...")
loader = AutoLoader("lm",
                    model_name="aquilacode-multi",
                    use_cache=True,
                    fp16=True,
                    device=device,
                    model_dir=model_dir)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.half()
model.eval()
model.to(device)

vocab = tokenizer.get_vocab()

id2word = {v: k for k, v in vocab.items()}
predictor = Predictor(model, tokenizer)

max_new_tokens = 256

texts = ["#补全代码\ndef quick_sort(x):"]

for text in texts:
    input_ids = tokenizer.encode_plus_non_glm(text)["input_ids"][:-1]
    input_length = len(input_ids)

    max_length = input_length + max_new_tokens
    with torch.no_grad():
        res = predictor.predict_generate_randomsample(
            text, out_max_length=max_length, top_p=0.95, temperature=0.1)
        print(res)
