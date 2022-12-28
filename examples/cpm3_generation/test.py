#coding:utf-8

import torch
import os
import json
from flagai.model.cpm3_model import CPM3Config, CPM3
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer.cpm_3 import CPM3Tokenizer
from tqdm import tqdm

from arguments import get_args
from generation import generate

def get_tokenizer():
    AutoLoader("lm", "cpm3", model_dir="./checkpoint")
    tokenizer = CPM3Tokenizer("./checkpoint/cpm3/vocab.txt", space_token='</_>', line_token='</n>', )
    return tokenizer


def get_model(vocab_size):
    config = CPM3Config.from_json_file("cpm3-large-32head.json")
    config.vocab_size = vocab_size
    print("vocab size:%d" % (vocab_size))

    model = CPM3(config)
    # if args.load != None:
    model.cuda()
    # if args.load != None:
    model.load_state_dict(
        torch.load("./checkpoint/cpm3/pytorch_model.pt"),
        strict=True
    )
    # else:
    #     bmp.init_parameters(model)
    return model


def setup_model():
    tokenizer = get_tokenizer()
    model = get_model(tokenizer.vocab_size)
    print("Model mem\n", torch.cuda.memory_summary())
    return tokenizer, model


def initialize():
    # get arguments
    args = get_args()
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def preprocess(sample):
    instance = {
        'mode': 'lm',
        'source': [sample, ""],
        'target': "",
        'control': {
            'keywords': [],
            'genre': '',
            'relations': [],
            'events': []
        }
    }
    return instance

def main():
    args = initialize()
    tokenizer, model = setup_model()
    input_sample = ["圣诞节送给男友这块商务手表，让男友倍有面！", 
        "蒸馏水是经过蒸馏和冷凝的水。其利用蒸馏设备使水蒸汽化，然后使水蒸气凝成水。", 
        "化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店，", "前段时间闺蜜来我家聚会。给我送了份超喜欢的礼物——玛莉娜1984的三件化妆品",
        "记得屈臣氏开业以来，我就很喜欢来这里购物，因为这里品牌好物多，品质好而且价格还亲民", 
        "春节营销具有很强的文化属性加成，一方面“办年货”习俗增强了消费意愿", 
        "女人都爱珠宝，宝石，钻石，翡翠，珍珠。水晶，说真的这个世界上好看又美好的事物太多啦。",
        "我觉得送化妆品类的礼物难度比较高，比如粉底不确定色号，眼影不知道喜不喜欢或者适不适合。", 
        "1.肉类：现在家庭饲养的猫咪基本上都是喂养猫干粮，有时间的话可以去超市购买一些肉类，切碎给猫咪吃，也可以制作成肉干、冻干，当作小零食。", 
        "今天出门居然忘记带口红！慌张！路过屈臣氏本想拿个试用装涂个颜色，结果被柜姐安利了新款口红。刚刚上市非常热乎！骨胶原口红，非常推荐！"]
    for line in input_sample:
        # sample = json.loads(line)
        instance = preprocess(line)
        target_span_len = args.span_length

        # 指定最短生成长度
        min_len = 2  # 确保生成内容不为空
        res_str = ""
        for it in generate(model, tokenizer, instance, target_span_len, beam=args.beam_size,
                           temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                           no_repeat_ngram_size=args.no_repeat_ngram_size, repetition_penalty=args.repetition_penalty,
                           random_sample=args.random_sample, min_len=min_len,
                           contrastive_search=args.use_contrastive_search):
            res_str += it

        template = {
            "story": res_str,
            "outline": instance['control']['relations'],
            "source": line
        }

        output_sample = json.dumps(template, ensure_ascii=False)
        print(output_sample)
     

if __name__ == "__main__":
    main()
