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

#from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp
from gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import update_graph_cache
from flagai.data.tokenizer import Tokenizer
import uvicorn, json, datetime
from asgiref.sync import sync_to_async
import requests
import re
from fastapi.responses import StreamingResponse


checkpoint_path = '/share/ldwang/checkpoints_pred/'
model_name = 'Aquila-7b-67000'
model_name = 'Aquila-7b-67000-sft-10m-convo-v2'

tokenizer = Tokenizer.from_pretrained("llama-30b-en", 
                                      cache_dir=os.path.join(checkpoint_path, model_name))
vocab = tokenizer.get_vocab()

id2word = {v:k for k, v in vocab.items()}


device = "cuda:0"
model_info = model_name
server_port = 5050
model_path = os.path.join(checkpoint_path, model_name)
model_path = os.path.join(model_path, 'pytorch_model.bin')

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
config.layer_norm_epsilon = 1e-5
print(config)
dtype = torch.float16

torch.cuda.set_device(device)

model = GPTLMHeadModel(config, 
                       device=device, 
                       dtype=dtype)

sd = torch.load(model_path, map_location="cpu")#["module"]

print(f"正在加载参数")
model.load_state_dict(sd, strict=True)
print(f"参数加载成功")

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

from conversation_convo_v2 import covert_prompt_to_input_ids, covert_prompt_to_input_ids_new


def predict(prompt, seed, max_length, topk, topp, t):
    set_random_seed(seed)
    
    # input_length = len(prompt) + len("[CLS]")
    # input_ids = tokenizer.encode_plus("I will introduce beijing, ")["input_ids"]
    #prompt = re.sub('\n+', '\n', prompt)
    #inputs = dict()
    #inputs['prompt'] = prompt
    #print(f"prompt {inputs}")
    #input_ids = covert_prompt_to_input_ids(prompt, tokenizer)
    input_ids = covert_prompt_to_input_ids_new(prompt, tokenizer)
    ## fix
    if input_ids[-1] == 221:
        input_ids = input_ids[:-1] #### TODO
    print(f"input_ids {input_ids}")

    # input_ids = tokenizer.encode_plus(prompt)["input_ids"]
    input_length = len(input_ids)

    input_ids = torch.tensor(input_ids)[None, ].to(device)

    with torch.no_grad():
        model_in = tokenizer.decode(input_ids[0].cpu().numpy())

        if max_length == 0:
            ## 计算每个token的预测概率，而不需要预测下一个了。
            #print(model(input_ids)[0].shape)
            #print(model(input_ids).shape)
            logits = model(input_ids)[0]
            logits = logits.softmax(dim=-1)
            # print(logits.shape)

            probs = []
            for index in range(1, input_ids.shape[1]):
                probs.append(logits[0, index-1, input_ids[0, index].item()].cpu().item())

            print(len(input_ids[0]))
            print(len(probs))

            tokens = input_ids[0].cpu().numpy().tolist()

            convert_tokens = []
            for t in tokens:
                if t == 100006:
                    convert_tokens.append("[CLS]")
                else :
                    convert_tokens.append(id2word.get(t, "[unkonwn_token]"))


            return "", convert_tokens, [0] + probs, model_in
    
        out = model.generate(input_ids=input_ids, max_length=input_length + max_length, top_k=topk,
                         vocab_size=config.vocab_size, fused_ft_kernel=True,
                         return_dict_in_generate=True, output_scores=True, timing=True,
                         eos_token_id=100007)

        print(out.keys())
        model_pred = tokenizer.decode(out.sequences[0].cpu().numpy()[input_length:])

        ### fix
        PREFIX_BOT = ' Assistant: '
        prefix_len = len(PREFIX_BOT)
        if model_pred[:prefix_len] == PREFIX_BOT:
            model_pred = model_pred[prefix_len:]
        
        raw_pred = model_pred

        convert_tokens = []
        ids = out.sequences[0].cpu().numpy()
        for t in ids:
            convert_tokens.append(id2word.get(t, "[unkonwn_token]"))

        convert_tokens = convert_tokens[input_length:]

        print(out.scores[0])

        probs = out.scores[0].cpu().numpy()
        probs = probs[input_length: ]

        print(convert_tokens)
        print(probs)
        
        if "###" in model_pred:
            special_index = model_pred.index("###")
            model_pred = model_pred[: special_index]
            token_length = len(tokenizer.encode_plus(model_pred)["input_ids"][1:-1])
            convert_tokens = convert_tokens[:token_length]
            probs = probs[:token_length]
        
        if "[UNK]" in model_pred:
            special_index = model_pred.index("[UNK]")
            model_pred = model_pred[:special_index]
            token_length = len(tokenizer.encode_plus(model_pred)["input_ids"][1:-1])
            convert_tokens = convert_tokens[:token_length]
            probs = probs[:token_length]
       
        if "</s>" in model_pred:
            special_index = model_pred.index("</s>")
            model_pred = model_pred[: special_index]
            token_length = len(tokenizer.encode_plus(model_pred)["input_ids"][1:-1])
            convert_tokens = convert_tokens[:token_length]
            probs = probs[:token_length]

        if model_pred[0] == " ":
            model_pred = model_pred[1:]
        
            convert_tokens = convert_tokens[1:]
            probs = probs[1:]

    return model_pred, convert_tokens, probs, model_in, raw_pred

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
        res, tokens, probs, model_in, raw_pred = await sync_to_async(predict)(contexts, seed, max_length=config['max_new_tokens'], topk=topk, topp=topp, t= t)

        result = {
            "completions": [{
                "text": res,
                "tokens": tokens,
                "logprobs": probs,
                "top_logprobs_dicts": [{k: v} for k, v in zip(tokens, probs)],
                "model_in": model_in,
                "raw_pred": raw_pred,
            }],
            "input_length": len(config["prompt"]),
            "model_info": model_info}

        return result
    
    @app.post("/sream_func")
    async def get_generate_h(request: Request):
        # config = json.loads(request.json)
        json_post_raw = await request.json()
        configs = json.loads(json_post_raw)

        print("request come in")
        contexts = configs["prompt"]
        contexts = re.sub('\n+', '\n', contexts)
        input_ids = tokenizer.encode_plus(contexts)["input_ids"]
        input_length = len(input_ids) - 1

        input_ids = torch.tensor(input_ids[:-1])[None, ].to(device)
        
        topk= configs.get("top_k_per_token", 100)
        topp = configs.get("top_p", 1.0)
        t = configs.get("temperature", 1.0)
        seed = configs.get("seed", 123)

        print(f"开始运算")
        fun = model.generate(input_ids=input_ids, max_length=input_length + configs['max_new_tokens'], top_k=topk,
                         vocab_size=config.vocab_size, fused_ft_kernel=True,
                         return_dict_in_generate=True, output_scores=True, timing=True,
                         eos_token_id=100007, stream=True, detokenizer=tokenizer)

        def trans():
            while True:
                try:
                    yield next(fun)
                    #yield next(tokenizer.decode(fun.cpu().numpy()))
                except Exception as e:
                    print(e)
                    break

        return StreamingResponse(trans(), media_type="text/plain")
 
    return app 

app = init_flask()

'''
ip = "http://120.92.208.64"

url_register = 'https://flagopen.baai.ac.cn:9443/api/set_model_info'
headers = {
    'Api-Token': 'aaaa',
    'Content-Type': 'application/json'
}
data = {
    'name': model_info,
    'info': {'url': ip + ':' + str(server_port+10) + '/sream_func', "stream": True}
}
response = requests.post(url_register, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.content.decode('utf-8'))
'''



uvicorn.run(app, host='0.0.0.0', port=server_port, workers=1)

'''
url_del = 'https://flagopen.baai.ac.cn:9443/api/del_model_info'

headers = {
    'Api-Token': 'aaaa',
    'Content-Type': 'application/json'
}

data = {
    'name': model_info
}

response = requests.post(url_del, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.content.decode('utf-8'))
'''


