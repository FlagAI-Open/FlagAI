
import torch
import os
import argparse
import sys
sys.path.append('../../../../flagai-internal-bmt-flashatten')
from flagai import mpu
from flagai.auto_model.auto_loader import AutoLoader
import random
import numpy as np
from flagai.model.predictor.predictor import Predictor
from pathlib import Path 
from flagai.data.tokenizer import Tokenizer
import time
import torch.distributed as dist
import json 
import uvicorn, json, datetime
from asgiref.sync import sync_to_async
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_info = "Aquila-7b-v1-sft-5m"
model_dir = "./state_dict/"
server_port = 5051
device = "cuda:0"

print(f"building model...")
loader = AutoLoader("lm", model_name="llama-7b-en",
                    only_download_config=False,
                    use_cache=True,
                    fp16=True,
                    model_dir=model_dir)

model = loader.get_model()
model.eval()
model.to(device)
tokenizer = Tokenizer.from_pretrained("llama-30b-en", 
                                      cache_dir="../../gpt2_new_100k/")
predictor = Predictor(model, tokenizer)
vocab = tokenizer.get_vocab()

id2word = {v:k for k, v in vocab.items()}

def predict(prompt, seed, max_length, topk, topp, t, candidate=None):
    set_random_seed(seed)
    
    with torch.no_grad():
        # prompt = "#用户#" + prompt + " " + "#ai助手#"
        #prompt = "[CLS]#用户#" + prompt + " " + "#ai助手#"
        # prompt = "[CLS]#用户#" + prompt + " " + "#ai助手#</s>"
        
        model_in = prompt

        out, tokens, probs, candidate_probs = predictor.predict_generate_randomsample(prompt, 
                                                        out_max_length=max_length, 
                                                        top_p=topp, 
                                                        temperature=t,
                                                        top_k=topk,
                                                        candidate=candidate
                                                        )
    
    convert_tokens = []
    for t in tokens:
        convert_tokens.append(id2word.get(t, "[unkonwn_token]"))
    #print(tokens)
    #print(probs)

    if candidate is not None:
        assert max_length == 1
        tokens = candidate_probs

    return out, convert_tokens, probs, model_in

def init_flask():
    from fastapi import FastAPI, Request

    app = FastAPI()

    @app.post("/func")
    async def get_generate_h(request: Request):
        # config = json.loads(request.json)
        json_post_raw = await request.json()
        config = json.loads(json_post_raw)

        print("request come in")
        contexts = config["prompt"]
        topk= config.get("top_k_per_token", 20)
        topp = config.get("top_p", 0.9)
        t = config.get("temperature", 0.9)
        seed = config.get("seed", 123)
        
        candidate = config.get("candidate", None)        
        print(f"开始运算")
        res, tokens, probs, model_in = await sync_to_async(predict)(contexts, seed, max_length=config['max_new_tokens'], topk=topk, topp=topp, t= t, candidate=candidate)

        result = {
            "completions": [{
                "text": res,
                "tokens": tokens,
                "logprobs": probs,
                "top_logprobs_dicts": [{k: v} for k, v in zip(tokens, probs)],
                "model_in": model_in
            }],
            "input_length": len(config["prompt"]),
            "model_info":model_info}

        return result

    return app

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

app = init_flask()

uvicorn.run(app, host='0.0.0.0', port=server_port, workers=1)
