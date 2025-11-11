# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comments]
# ------------------------------------------------------------------

import json,jsonlines

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

aquila_model_path = "./aquila-7b-sql-23-10-07-to-hf"


def generate_prompt(input: str):
    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input}###Assistant:"
    return prompt

def read_file(jsonl_data):
    conversations = []
    with jsonlines.open(jsonl_data) as reader:
        for line in reader:
            conversations.append(line)
    return conversations


tokenizer = AutoTokenizer.from_pretrained(aquila_model_path, trust_remote_code=True)
print("tokenizer.pad_token_id",tokenizer.pad_token_id,tokenizer.eos_token_id)


model = AutoModelForCausalLM.from_pretrained(
    aquila_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')

print("model load")
model.eval()
model.to("cuda:0")
torch.manual_seed(123)
# tokens = covert_prompt_to_input_ids_with_history(input, history=[], tokenizer=tokenizer, max_token=512)

stop_tokens = ["###", "[UNK]", "</s>","<|endoftext|>"]

src_file = "./data/infer_sample.txt"
contents = read_file(src_file)

answers = []
with torch.no_grad():
    for one in contents:
        question = one["prompt"]
        # question = ""
        _input = generate_prompt(question)
        tokens = tokenizer.encode_plus(_input, None, max_length=None)['input_ids']
        tokens = torch.tensor(tokens)[None,].to("cuda:0")
        out = model.generate(tokens, do_sample=False, max_length=1024, eos_token_id=100007,max_new_tokens=512,
                                bad_words_ids=[[tokenizer.encode(token)[0] for token in stop_tokens]])[0]
        out = tokenizer.decode(out.cpu().numpy().tolist())
        if "###Assistant:" in out:
            special_indx = out.index("###Assistant")
            out = out[special_indx+len("###Assistant"):]
        if "</s>" in out:
            special_indx = out.index("</s>")
            out = out[:special_indx]

        msg = {"qestion": question, "ans": out}
        print("msg",msg)
        info = json.dumps(msg, ensure_ascii=False)
