# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.data.tokenizer import Tokenizer
import torch.nn as nn
from flagai.model.predictor.aquila import aquila_generate


state_dict = "./checkpoints_in/"
model_name = 'aquilachat-7b'  


loader = AutoLoader("lm",
                    model_dir=state_dict,
                    model_name=model_name,
                    use_cache=True,
                    fp16=True,
                    device='cuda',
                    adapter_dir='/data2/yzd/FlagAI/examples/Aquila/Aquila-chat/checkpoints_out/aquilachat_experiment/2023081313/') # Directory to adapter_model.bin and adapter_config.json
model = loader.get_model()

tokenizer = loader.get_tokenizer()
model.eval()
model.cuda()

predictor = Predictor(model, tokenizer)

texts = [
    "Find the product of the numbers: 5 and 8",
    "Create a list of potential topics for a company newsletter",
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a dragon and a knight.",
    "翻译成英文: '我饿了想吃饭'",
    "write a fairy tale for me",
    "用英文回答: 世界上最高的地方在哪里?"
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
