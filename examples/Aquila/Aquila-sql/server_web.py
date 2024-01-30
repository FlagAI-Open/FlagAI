# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
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
import uvicorn
from asgiref.sync import sync_to_async
from fastapi.responses import StreamingResponse

import jsonlines
import torch
import tqdm
project_path = "../../../../FlagAI"
sys.path.append(project_path)

from cyg_conversation import covert_prompt_to_input_ids_with_history
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.aquila_server import aquila_generate_by_ids,aquila_generate_by_ids_stream


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

state_dict = sys.argv[1]


server_port = 9175

model_name = 'aquilachat-7b'

device = f"cuda:0"
print(f"device is {device}")


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




def init_flask():
    from fastapi import FastAPI, Request

    app = FastAPI()

    @app.post("/func")
    async def get_generate_h(request: Request):
        json_post_raw = await request.json()
        config = json.loads(json_post_raw)

        text = config["prompt"]
        topp = config.get("top_p", 0.95)
        max_length = config.get("max_new_tokens", 256)
        topk = config.get("top_k_per_token", 1000)
        temperature = config.get("temperature", 0.9)
        sft = config.get("sft", False)
        seed = config.get("seed", 1234)
        history = config.get("history", [])

        tokens = covert_prompt_to_input_ids_with_history(text, history, tokenizer, max_length)

        print(f"sft is {sft}")

        out, tokens, probs, model_in = await sync_to_async(predict)(tokenizer, model, text,
                                           max_gen_len=max_length, top_p=topp,
                                           prompts_tokens=tokens, topk=topk,
                                           temperature=temperature, sft=sft, seed=seed)


        print(f"que is: [{text}]")
        print(f"ans is: [{out}]")
        out = out.replace("sql","")
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

    @app.post("/stream_func")
    async def get_generate_stream(request: Request):
        json_post_raw = await request.json()
        config = json.loads(json_post_raw)

        text = config["prompt"]
        topp = config.get("top_p", 0.95)
        max_length = config.get("max_new_tokens", 256)
        topk = config.get("top_k_per_token", 1000)
        temperature = config.get("temperature", 0.9)
        sft = config.get("sft", False)
        seed = config.get("seed", 1234)
        history = config.get("history", [])
        history = []

        tokens = covert_prompt_to_input_ids_with_history(text, history, tokenizer, max_length)

        print(f"sft is {sft}")

        with torch.no_grad():
            fun = aquila_generate_by_ids_stream(model, tokenizer, tokens,
                                                out_max_length=max_length+len(tokens),
                                                top_k=topk, top_p=topp,
                                                temperature=temperature,
                                                seed=seed, device=device)

        gene_time = 15
        def trans():
            start_time = time.time()
            while True:
                try:
                    next_token = next(fun)
                    if "sql" in next_token:
                        next_token = next_token.replace("sql","")
                    logger.info(f"chatmodel next token is: {next_token}")
                    yield next_token
                except StopIteration:
                    logger.info("get StopIteration")
                    break
                except Exception as e:
                    logger.info(traceback.print_exc())
                    pass
                if time.time() - start_time > gene_time:
                    print("time up")
                    break

        return StreamingResponse(trans(), media_type="text/plain")

    return app

app = init_flask()

uvicorn.run(app, host='0.0.0.0', port=server_port, workers=1)
