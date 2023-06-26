# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.aquila import aquila_generate
from flagai.data.tokenizer import Tokenizer
import bminf

state_dict = "./checkpoints_in"
model_name = 'aquilachat-7b'

loader = AutoLoader("lm",
                    model_dir=state_dict,
                    model_name=model_name,
                    use_cache=True,
                    fp16=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()
cache_dir = os.path.join(state_dict, model_name)

model.eval()

with torch.cuda.device(0):
    model = bminf.wrapper(model, quantization=False, memory_limit=2 << 30)  # n << 30 is equivalent to n GB memory limit

predictor = Predictor(model, tokenizer)

texts = [
    "北京为什么是中国的首都？",
    "1+1=",
    "为什么湘菜那么甜？",
    "东三省和海南岛的区别？",
]

for text in texts:
    print('-' * 80)
    print(f"text is {text}")

    from cyg_conversation import default_conversation

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)

    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}",
                                   None,
                                   max_length=None)['input_ids']
    tokens = tokens[1:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer,
                              model, [text],
                              max_gen_len := 200,
                              top_p=0.95,
                              prompts_tokens=[tokens])
        print(f"pred is {out}")
