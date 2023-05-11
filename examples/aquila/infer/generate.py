
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_info = "Aquila-7b-v1-sft-5m"
model_dir = "../state_dict/"
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

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

texts = [
        "拥有美丽身材是大多数女人追求的梦想，甚至有不少mm为了实现这个梦而心甘情愿付出各种代价，",
        "2007年乔布斯向人们展示iPhone并宣称它将会改变世界",
        "从前有座山，",
        "如何摆脱无效焦虑?",
        "北京在哪儿?",
        #"北京",
        "汽车EDR是什么",
        "My favorite animal is",
        "今天天气不错",
        "如何评价许嵩?",
        "汽车EDR是什么",
        "给妈妈送生日礼物，怎么选好？",
        "1加1等于18497是正确的吗？",
        "如何给汽车换胎？",
        "以初春、黄山为题，做一首诗。",
        "What is machine learning?",
        "Machine learning is",
        "Nigerian billionaire Aliko Dangote says he is planning a bid to buy the UK Premier League football club.",
        "The capital of Germany is the city of ",
        ]


for text in texts:
    out, convert_tokens, probs, model_in = predict(text, 123, 138, 100, 1.0, 1.0, candidate=None)
    print(text)
    print(out)
