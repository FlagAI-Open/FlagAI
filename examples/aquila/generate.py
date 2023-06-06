import os
import torch
import sys;sys.path.append("/data/yzd/FlagAI")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.llama import llama_generate
from flagai.data.tokenizer import Tokenizer
# import bminf

state_dict = "./checkpoints"
model_name = 'aquilachat-7b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True)
model = loader.get_model()

cache_dir = os.path.join(state_dict, model_name)
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#print('*'*20, "tokenizer", tokenizer)

model.eval()
model.half()
model.cuda()
# with torch.cuda.device(0):
#     model = bminf.wrapper(model, quantization=False, memory_limit=2 << 30)

predictor = Predictor(model, tokenizer)


texts = [
        #"I am ",
        #"1月7日，五华区召开“中共昆明市五华区委十届三次全体(扩大)会议”，",
        #"1月7日，五华区召开“中共昆明市五华区委十届三次全体(扩大)会议”，区委书记金幼和作了《深入学习贯彻党的十八大精神，奋力开创五华跨越发展新局面》的工作报告。",
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
        #"Machine learning is",
        #"Nigerian billionaire Aliko Dangote says he is planning a bid to buy the UK Premier League football club.",
        #"The capital of Germany is the city of ",
        ]

texts = [
        "北京为什么是中国的首都？",
        "1+1=",
        "为什么湘菜那么甜？",
        "东三省和海南岛的区别？",
        ]
## 
def pack_obj(text):
    obj = dict()
    obj['id'] = 'demo'

    obj['conversations'] = []
    human = dict()
    human['from'] = 'human'
    human['value'] = text
    obj['conversations'].append(human)
    # dummy bot
    bot = dict()
    bot['from'] = 'gpt'
    bot['value'] = ''
    obj['conversations'].append(bot)

    obj['instruction'] = ''

    return obj

def delete_last_bot_end_singal(convo_obj):
    conversations = convo_obj['conversations']
    assert len(conversations) > 0 and len(conversations) % 2 == 0
    assert conversations[0]['from'] == 'human'

    last_bot = conversations[len(conversations)-1]
    assert last_bot['from'] == 'gpt'

    ## from _add_speaker_and_signal
    END_SIGNAL = "\n"
    len_end_singal = len(END_SIGNAL)
    len_last_bot_value = len(last_bot['value'])
    last_bot['value'] = last_bot['value'][:len_last_bot_value-len_end_singal]
    return

def convo_tokenize(convo_obj, tokenizer):
    chat_desc = convo_obj['chat_desc']
    instruction = convo_obj['instruction']
    conversations = convo_obj['conversations']
            
    # chat_desc
    example = tokenizer.encode_plus(f"{chat_desc}", None, max_length=None)['input_ids']
    EOS_TOKEN = example[-1]
    example = example[:-1] # remove eos
    # instruction
    instruction = tokenizer.encode_plus(f"{instruction}", None, max_length=None)['input_ids']
    instruction = instruction[1:-1] # remove bos & eos
    example += instruction

    for conversation in conversations:
        role = conversation['from']
        content = conversation['value']
        print(f"role {role}, raw content {content}")
        content = tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
        content = content[1:-1] # remove bos & eos
        print(f"role {role}, content {content}")
        example += content
    return example

for text in texts:
    print('-'*80)
    print(f"text is {text}")

    from examples.gpt3_pretrain.llama.ym_conversation import default_conversation

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)

    #print(conv.get_prompt())
    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
    tokens = tokens[1:-1]
    #print(f"tokens \n {tokens}")

    with torch.no_grad():
        #out = predictor.predict_generate_randomsample(text, out_max_length=200,top_p=0.95)
        #out = predictor.predict_generate_randomsample(text, out_max_length=200, temperature=0)
        #out = llama_generate(tokenizer, model, [text], max_gen_len:=200, temperature=0, prompts_tokens=[tokens])
        out = llama_generate(tokenizer, model, [text], max_gen_len:=200, top_p=0.95, prompts_tokens=[tokens])
        print(f"pred is {out}")

'''
for text in texts:
    print('-'*80)
    #text = f'#用户#{text} #ai助手#' #sft
    #text = f'{text}' #base
    print(f"text is {text}")
    #out = predictor.predict_generate_randomsample(text, out_max_length=200,top_p=0.95)
    with torch.no_grad():
        out = predictor.predict_generate_randomsample(text, out_max_length=200, temperature=0)
        print(f"pred is {out}")
'''





# import torch
# import os
# import argparse
# import sys
# # sys.path.append('../../../../flagai-internal-bmt-flashatten')
# import sys;sys.path.append("/data/yzd/FlagAI")
# from flagai import mpu
# from flagai.auto_model.auto_loader import AutoLoader
# import random
# import numpy as np
# from flagai.model.predictor.predictor import Predictor
# from pathlib import Path 
# from flagai.data.tokenizer import Tokenizer
# import time
# import torch.distributed as dist
# import json 
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# model_info = "Aquila-7b-v1-sft-5m"
# model_dir = "./checkpoints"
# server_port = 5051
# device = "cuda:0"

# print(f"building model...")
# loader = AutoLoader("lm", model_name="aquilachat-7b",
#                     only_download_config=False,
#                     use_cache=True,
#                     fp16=True,
#                     model_dir=model_dir)

# model = loader.get_model()
# model.eval()
# model.to(device)
# tokenizer = Tokenizer.from_pretrained("llama-30b-en", 
#                                       cache_dir="/data/yzd/FlagAI/examples/gpt3_pretrain/gpt2_new_100k")

# predictor = Predictor(model, tokenizer)
# vocab = tokenizer.get_vocab()

# id2word = {v:k for k, v in vocab.items()}

# def predict(prompt, seed, max_length, topk, topp, t, candidate=None):
#     set_random_seed(seed)
    
#     with torch.no_grad():
#         # prompt = "#用户#" + prompt + " " + "#ai助手#"
#         #prompt = "[CLS]#用户#" + prompt + " " + "#ai助手#"
#         # prompt = "[CLS]#用户#" + prompt + " " + "#ai助手#</s>"
        
#         model_in = prompt

#         out, tokens, probs, candidate_probs = predictor.predict_generate_randomsample(prompt, 
#                                                         out_max_length=max_length, 
#                                                         top_p=topp, 
#                                                         temperature=t,
#                                                         top_k=topk,
#                                                         prompts_tokens=candidate
#                                                         )
    
#     convert_tokens = []
#     for t in tokens:
#         convert_tokens.append(id2word.get(t, "[unkonwn_token]"))
#     #print(tokens)
#     #print(probs)

#     if candidate is not None:
#         assert max_length == 1
#         tokens = candidate_probs

#     return out, convert_tokens, probs, model_in

# def set_random_seed(seed):
#     """Set random seed for reproducability."""
#     if seed is not None and seed > 0:
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

# texts = [
#         "拥有美丽身材是大多数女人追求的梦想，甚至有不少mm为了实现这个梦而心甘情愿付出各种代价，",
#         "2007年乔布斯向人们展示iPhone并宣称它将会改变世界",
#         "从前有座山，",
#         "如何摆脱无效焦虑?",
#         "北京在哪儿?",
#         #"北京",
#         "汽车EDR是什么",
#         "My favorite animal is",
#         "今天天气不错",
#         "如何评价许嵩?",
#         "汽车EDR是什么",
#         "给妈妈送生日礼物，怎么选好？",
#         "1加1等于18497是正确的吗？",
#         "如何给汽车换胎？",
#         "以初春、黄山为题，做一首诗。",
#         "What is machine learning?",
#         "Machine learning is",
#         "Nigerian billionaire Aliko Dangote says he is planning a bid to buy the UK Premier League football club.",
#         "The capital of Germany is the city of ",
#         ]


# for text in texts:
#     out, convert_tokens, probs, model_in = predict(text, 123, 138, 100, 1.0, 1.0, candidate=None)
#     print(text)
#     print(out)
