#coding:utf-8

from email.policy import default
from operator import length_hint
import time
import random
import torch
import numpy as np
import os
import json
from flagai.model.cpm3_model import CPM3Config, CPM3
from flagai.data.tokenizer.cpm_3 import CPM3Tokenizer

from arguments import get_args
from generation import generate

import random

def get_tokenizer(args):
    tokenizer = CPM3Tokenizer(args.vocab_file, space_token = '</_>', line_token = '</n>',)
    return tokenizer

def get_model(args, vocab_size):
    config = CPM3Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))


    model = CPM3(config)
    # if args.load != None:
    model.cuda()
    # if args.load != None:
    model.load_state_dict(
        torch.load(args.load),
        strict = True
    )
    # else:
    #     bmp.init_parameters(model)
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer.vocab_size)
    print("Model mem\n", torch.cuda.memory_summary())
    return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def random_mask(source: str):
    if type(source) == list:
        return source
    length = len(source)
    half = length // 2
    res = [source[:half]]
    return res



def main():
    args = initialize()
    tokenizer, model = setup_model(args)

    fout = open("{}".format(args.output_file), "w", encoding="utf-8")
    fin = open('{}'.format(args.input_file), 'r', encoding='utf-8')

    for line in fin:
        instance = {
            'mode': 'lm',
            'source': [],
            'target': "",
            'control': {
                'keywords': [],
                'genre': "",
                'relations': [],
                'events': []
            }
        }
        res = json.loads(line)
        for key in instance.keys():
            if key == 'source':
                instance[key] = random_mask(res.get(key, instance[key]))
            else:
                instance[key] = res.get(key, instance[key])
        target_span_len = args.span_length

        # 指定最短生成长度
        min_len = 2 # 确保生成内容不为空
        for it in generate(model, tokenizer, instance, target_span_len, beam=args.beam_size,
                            temperature = args.temperature, top_k = args.top_k, top_p = args.top_p,
                            no_repeat_ngram_size = args.no_repeat_ngram_size, repetition_penalty = args.repetition_penalty, 
                            random_sample=args.random_sample, min_len=min_len, contrastive_search=args.use_contrastive_search):
            
            fout.write(it)
            fout.flush()
        fout.write('\n')
    
    fin.close()
    fout.close()

if __name__ == "__main__":
    main()
