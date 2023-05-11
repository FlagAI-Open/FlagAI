# Copyright (c) 2023, Tri Dao.

# To run the huggingface implementation, we first need to convert the weights:
# https://github.com/huggingface/transformers/pull/21955
# python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir $CHECKPOINT_DIR/llama --model_size 7B --output_dir $CHECKPOINT_DIR$/llama/7B-hf
# and repeat for 13B, 30B, 65B

import os
import time
from pathlib import Path
current_dir = Path(__file__).parent.absolute()
import random 
import numpy as np 
import torch
import sys
sys.path.append('../../../../flagai-internal-bmt-flashatten')
from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import update_graph_cache
from flagai.data.tokenizer import Tokenizer
import uvicorn, json, datetime
from asgiref.sync import sync_to_async


tokenizer = Tokenizer.from_pretrained("llama-30b-en", 
                                      cache_dir="../../gpt2_new_100k/")
vocab = tokenizer.get_vocab()

id2word = {v:k for k, v in vocab.items()}
ckpt_iter = int(sys.argv[1])
model_name='Aquila-7b'
device = "cuda:0"
model_info = f"model-{ckpt_iter}"
server_port = 5050 
model_path = f"{ckpt_iter}-flash/pytorch_model.bin"

checkpoint_path = './state_dict'
config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
config.vocab_size=100008
config.use_cache = True
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.fused_bias_fc = False
config.use_flash_attn = False
config.fused_mlp = False  # We don't have fused GatedMLP yet
config.fused_dropout_add_ln = False
config.residual_in_fp32 = False

print(config)
dtype = torch.float16

torch.cuda.set_device(device)

model = GPTLMHeadModel(config, 
                       device=device, 
                       dtype=dtype)

sd = torch.load(model_path, map_location="cpu")

print(f"正在加载参数")
model.load_state_dict(sd, strict=False)
print(f"参数加载成功")

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def predict(prompt, seed, max_length, topk, topp, t):
    set_random_seed(seed)
    
    input_ids = tokenizer.encode_plus(prompt)["input_ids"]
    input_length = len(input_ids) - 1

    input_ids = torch.tensor(input_ids[:-1])[None, ].to(device)

    with torch.no_grad():

        out = model.generate(input_ids=input_ids, max_length=input_length + max_length, top_k=topk,
                         vocab_size=config.vocab_size, fused_ft_kernel=True,
                         return_dict_in_generate=True, output_scores=True, timing=True,
                         eos_token_id=100007)

        model_pred = tokenizer.decode(out.sequences[0].cpu().numpy()[input_length:])

        convert_tokens = []
        ids = out.sequences[0].cpu().numpy()
        for t in ids:
            convert_tokens.append(id2word.get(t, "[unkonwn_token]"))

        convert_tokens = convert_tokens[input_length:]


        probs = out.scores[0].cpu().numpy()
        probs = probs[input_length: ]

    return model_pred, convert_tokens, probs

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
        topk= config.get("top_k_per_token", 100)
        topp = config.get("top_p", 1.0)
        t = config.get("temperature", 1.0)
        seed = config.get("seed", 123)

        print(f"开始运算")
        res, tokens, probs = await sync_to_async(predict)(contexts, seed, max_length=config['max_new_tokens'], topk=topk, topp=topp, t= t)

        result = {
            "completions": [{
                "text": res,
                "tokens": tokens,
                "logprobs": probs,
                "top_logprobs_dicts": [{k: v} for k, v in zip(tokens, probs)],
            }],
            "input_length": len(config["prompt"]),
            "model_info": model_info}

        return result
 
    return app 

app = init_flask()

uvicorn.run(app, host='0.0.0.0', port=server_port, workers=1)
