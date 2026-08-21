# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [CreatedDate]  : Tuesday, 2023-07-04 09:19:30
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comments]
# ------------------------------------------------------------------

import json
import logging
import os
import re
import sys
import time
import traceback

import jsonlines
import torch
import tqdm
project_path = "../../../../FlagAI"
sys.path.append(project_path)

from cyg_conversation import covert_prompt_to_input_ids_with_history
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.aquila_server import aquila_generate_by_ids



# NOTE: fork from FlagAI/examples/Aquila/Aquila-server/aquila_server.py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_LogFormat = logging.Formatter(
    "%(asctime)2s -%(name)-12s: %(levelname)-s/Line[%(lineno)d]/Thread[%(thread)d]  - %(message)s")

# create console handler with a higher log level
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(_LogFormat)
logger.addHandler(console)


filemaps = {
    "cspider": ["./data_collection/llm_input/sql_generate_from_cspider_dev_query.txt",
                "./data_collection/data/CSpider/dev_gold.sql"],


}

dataset_name = sys.argv[1]

state_dict = sys.argv[2]


server_port = 9173
dest_name = "_".join(state_dict.split("/")[-2:])


print("dest_name", dest_name)
model_name = 'aquilachat-7b'

device = f"cuda:0"
print(f"device is {device}")


def predict(tokenizer, model, text,
            max_gen_len=200, top_p=0.95,
            prompts_tokens=[], seed=1234, topk=100,
            temperature=0.9, sft=True):

    prompt = re.sub('\n+', '\n', text)
    if not sft:
        prompts_tokens = tokenizer.encode_plus(prompt)["input_ids"][:-1]

    model_in = tokenizer.decode(prompts_tokens)
    start_time = time.time()
    with torch.cuda.device(0):
        with torch.no_grad():
            out, tokens, probs = aquila_generate_by_ids(model=model, tokenizer=tokenizer,
                                                        input_ids=prompts_tokens,
                                                        out_max_length=max_gen_len, top_p=top_p, top_k=topk,
                                                        seed=seed, temperature=temperature, device=device)
    convert_tokens = []
    for t in tokens:
        if t == 100006:
            convert_tokens.append("[CLS]")
        else:
            convert_tokens.append(id2word.get(t, "[unkonwn_token]"))

    if "###" in out:
        special_index = out.index("###")
        out = out[: special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"][1:-1])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if "[UNK]" in out:
        special_index = out.index("[UNK]")
        out = out[:special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"][1:-1])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if "</s>" in out:
        special_index = out.index("</s>")
        out = out[: special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"][1:-1])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if len(out) > 0 and out[0] == " ":
        out = out[1:]

        convert_tokens = convert_tokens[1:]
        probs = probs[1:]

    return out, convert_tokens, probs, model_in


def get_generate_h(config):
    # print("request come in")
    text = config["prompt"]
    topp = config.get("top_p", 0.95)
    max_length = config.get("max_new_tokens", 256)
    topk = config.get("top_k_per_token", 1000)
    temperature = config.get("temperature", 0.9)
    sft = config.get("sft", False)
    seed = config.get("seed", 1234)
    history = config.get("history", [])

    tokens = covert_prompt_to_input_ids_with_history(text, history, tokenizer, max_length)

    out, tokens, probs, model_in = predict(tokenizer, model, text,
                                           max_gen_len=max_length, top_p=topp,
                                           prompts_tokens=tokens, topk=topk,
                                           temperature=temperature, sft=sft, seed=seed)

    result = {
        "completions": [{
            "text": out,
            "tokens": tokens,
            "logprobs": probs,
            "top_logprobs_dicts": [{k: v} for k, v in zip(tokens, probs)],
            "model_in": model_in
        }],
        "input_length": len(config["prompt"]),
        "model_info": model_name}

    return result


print(f"building model...")
start_time = time.time()
# NOTE: 这里注意加上device这个变量，否则在cpu中加载，很慢
loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True,
    fp16=True,
    device=device)
end_time = time.time()
print("Load model time:", end_time - start_time, "s", flush=True)

model = loader.get_model()
tokenizer = loader.get_tokenizer()

vocab = tokenizer.get_vocab()

id2word = {v: k for k, v in vocab.items()}

model.eval()
model.to(device)


def read_file(jsonl_data):
    conversations = []
    with jsonlines.open(jsonl_data) as reader:
        for line in reader:
            conversations.append(line)
    return conversations


src_file = filemaps[dataset_name][0]
raw_gold_file = filemaps[dataset_name][1]

dest_dir = f"../evaluation_result/{dest_name}_eval"
print("dest_dir", dest_dir)
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)


raw_pred_file = f"../evaluation_result/{dest_name}_eval/{dataset_name}_raw_pred.jsonl"
pred_file = f"../evaluation_result/{dest_name}_eval/{dataset_name}_predict.txt"
gold_file = f"../evaluation_result/{dest_name}_eval/{dataset_name}_gold.txt"

record_file = f"../evaluation_result/record.txt"

info_to_record = {
    "model_info": {
        "model_dir": state_dict,
        "frame": "flagai"
    },
    "evaluate_data": dataset_name,
    "file_info": {
        "src_file": src_file,
        "raw_gold_file": raw_gold_file,
        "raw_pred_file": raw_pred_file,
        "pred_file": pred_file,
        "gold_file": gold_file},

}

print("info_to_record", info_to_record)


def find_sql(text):
    pattern = r"```sql\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    # print("matches",matches)
    # for match in matches:
    #     print(match)
    return matches


if not os.path.exists(raw_pred_file):
    contents = read_file(src_file)
    raw_result = []
    for one in tqdm.tqdm(contents):
        inp_json = {
            "prompt": one["prompt"],
            # "sft":True
        }
        result = get_generate_h(inp_json)
        text = result["completions"][0]["text"]
        raw_result.append(json.dumps({"prompt": one["prompt"], "result": text}, ensure_ascii=False))

    with open(raw_pred_file, "w") as f:
        for one in raw_result:
            f.write(f"{one}\n")

else:
    raw_result = read_file(raw_pred_file)


#  读取ground truth信息
gold_contents = []
with open(raw_gold_file, "r") as fg:
    for line in fg:
        gold_contents.append(line.strip())

assert len(raw_result) == len(gold_contents)


fp = open(pred_file, "w")
fg_processed = open(gold_file, "w")
total_num = 0
valid_num = 0
for index in tqdm.tqdm(list(range(len(raw_result)))):
    total_num += 1
    pred_raw = raw_result[index]
    gold_raw = gold_contents[index].strip()

    if isinstance(pred_raw, str):
        pred_raw = json.loads(pred_raw)

    value = pred_raw["result"]
    sql_strs = find_sql(value)
    if sql_strs and sql_strs[0]:
        valid_num += 1
        _sql = sql_strs[0].replace('\n', ' ').strip()
        fp.write(f"{_sql}\n")
        fg_processed.write(f"{gold_raw}\n")

fp.close()
fg_processed.close()



print(f"total_num: [{total_num}]")
print(f"valid_num: [{valid_num}]")

info_to_record.update({
    "num": {
        "total_num": total_num,
        "valid_num": valid_num,
    }
})
with open(record_file, "a") as f:
    f.write(json.dumps(info_to_record, ensure_ascii=False)+"\n")


print(f"dest predict : {pred_file}")
print(f"dest gold  : {gold_file}")

