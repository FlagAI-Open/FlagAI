# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
import transformers

state_dict = "./checkpoints_in/"
model_name = 'Llama-3.1-8B'

loader = AutoLoader("llama3",
                    model_dir=state_dict,
                    model_name=model_name,
                    device='cuda',
                    use_cache=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()

model.eval()

model.cuda()

print("model loaded")

text = "Gravity is "

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("content:", content)
