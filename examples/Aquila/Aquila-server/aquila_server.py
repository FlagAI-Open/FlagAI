import torch
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np
from flagai.model.predictor.predictor import Predictor
from pathlib import Path
from flagai.data.tokenizer import Tokenizer
import json
import uvicorn, json
from asgiref.sync import sync_to_async
from cyg_conversation import covert_prompt_to_input_ids_with_history
from flagai.model.predictor.aquila_server import aquila_generate_by_ids, aquila_generate_by_ids_stream
from fastapi.responses import StreamingResponse
import time
import re
import sys

state_dict = "./checkpoints_in"
model_name = 'aquilachat-7b'
server_port = 7860

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
    start_time=time.time()
    with torch.cuda.device(int(sys.argv[1])):
        with torch.no_grad():
            out, tokens, probs = aquila_generate_by_ids(model=model, tokenizer=tokenizer,
                                    input_ids=prompts_tokens,
                                    out_max_length=max_gen_len, top_p=top_p, top_k=topk,
                                    seed=seed, temperature=temperature, device=device)
    convert_tokens = []
    for t in tokens:
        if t == 100006:
            convert_tokens.append("[CLS]")
        else :
            convert_tokens.append(id2word.get(t, "[unkonwn_token]"))

    if "###" in out:
        special_index = out.index("###")
        out = out[: special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"][1:-1])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    return out, convert_tokens, probs, model_in


def init_flask():
    from fastapi import FastAPI, Request

    app = FastAPI()

    @app.post("/func")
    async def get_generate_h(request: Request):
        json_post_raw = await request.json()
        config = json.loads(json_post_raw)

        print("request come in")
        text = config["prompt"]
        topp = config.get("top_p", 0.95)
        max_length = config.get("max_new_tokens", 256)
        topk = config.get("top_k_per_token", 1000)
        temperature = config.get("temperature", 0.9)
        history = config.get("history", [])
        sft = config.get("sft", False)
        seed = config.get("seed", 1234)

        tokens = covert_prompt_to_input_ids_with_history(text, history, tokenizer, max_length)

        out, tokens, probs, model_in = await sync_to_async(predict)(tokenizer, model, text,
                                           max_gen_len=max_length, top_p=topp,
                                           prompts_tokens=tokens, topk=topk,
                                           temperature=temperature, sft=sft,seed=seed)

        result = {
            "completions": [{
                "text": out,
                "tokens": tokens,
                "logprobs": probs,
                "top_logprobs_dicts": [{k: v} for k, v in zip(tokens, probs)],
                "model_in": model_in
            }],
            "input_length": len(config["prompt"]),
            "model_info":model_name}

        return result

    @app.post("/stream_func")
    async def get_generate_stream(request: Request):
        json_post_raw = await request.json()
        config = json.loads(json_post_raw)

        contexts = config["prompt"]
        topk= config.get("top_k_per_token", 20)
        topp = config.get("top_p", 0.9)
        t = config.get("temperature", 0.9)
        seed = config.get("seed", 1234)
        history = config.get("history", [])
        max_length = config.get("max_new_tokens", 256)
        gene_time = config.get("time", 15)
        gene_time = 40
        tokens = covert_prompt_to_input_ids_with_history(contexts, history, tokenizer, max_length)

        with torch.no_grad():
            fun = aquila_generate_by_ids_stream(model, tokenizer, tokens,
                                                out_max_length=max_length,
                                                 top_k=topk, top_p=topp,
                                                  temperature=t,
                                                   seed=seed, device=device )

        def trans():
            start_time = time.time()
            while True:
                try:
                    yield next(fun)
                except Exception as e:
                    print(e)
                    break
                if time.time() - start_time > gene_time:
                    print("time up")
                    break

        return StreamingResponse(trans(), media_type="text/plain")

    return app

print(f"building model...")
start_time = time.time()
loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True,
    fp16=True)
end_time = time.time()
print("Load model time:", end_time - start_time, "s", flush=True)

model = loader.get_model()
tokenizer = loader.get_tokenizer()

vocab = tokenizer.get_vocab()

id2word = {v:k for k, v in vocab.items()}

model.eval()
model.to(device)

predictor = Predictor(model, tokenizer)

app = init_flask()

uvicorn.run(app, host='0.0.0.0', port=server_port, workers=1)
