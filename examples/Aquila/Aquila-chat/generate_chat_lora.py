# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.data.tokenizer import Tokenizer
import torch.nn as nn
from flagai.model.tools.lora.prepare_lora import lora_transfer
state_dict = "./checkpoints_in/"
model_name = 'aquila-7b'


loader = AutoLoader("lm",
                    model_dir=state_dict,
                    model_name=model_name,
                    use_cache=True,
                    fp16=True,
                    adapter_dir='./output')
model = loader.get_model()

tokenizer = loader.get_tokenizer()

model.eval()
mode.half()
model.cuda()

predictor = Predictor(model, tokenizer)

texts = [
    "Find the product of the numbers: 5 and 8",
    "Provide five tips for effectively using tape measures",
    "Create a resume for a job in web development.",
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
    ## TODO for few-shot inference using plain text as inputs will get better results.
    ## tokens = tokenizer.encode_plus(f"{text}", None, max_length=None)['input_ids']
    tokens = tokens[1:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer,
                              model, [text],
                              max_gen_len := 200,
                              top_p=0.95,
                              prompts_tokens=[tokens])
        print(f"pred is {out}")