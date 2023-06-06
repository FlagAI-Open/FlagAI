# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

import bminf

if __name__ == '__main__':
    model_dir = "./"

    '''
    loader = AutoLoader("seq2seq",
                        "gpt2-base-en",
                        model_dir=model_dir)
    '''

    loader = AutoLoader(
        "seq2seq",
        model_name="gpm-13b",
        model_dir=model_dir,
    )

    model = loader.get_model()
    #model.cuda()
    model.half()
    with torch.cuda.device(1):
        model = bminf.wrapper(model, quantization=False, memory_limit=20 << 30)
    tokenizer = loader.get_tokenizer()
    predictor = Predictor(model, tokenizer)

    text = "Nigerian billionaire Aliko Dangote says he is planning a bid to buy the UK Premier League football club. "
    text = "Major League Baseball All-Century Team In 1999, the Major League Baseball"
    text = "Hollym Gate railway station"
    text = "Molly Henderson Molly Henderson (born September 14, 1953) is a former Commissioner of Lancaster County, Pennsylvania. The Commissioners are"
    text = "What is machine learning?"
    text = "Machine learning is"

    #print(f"inp is {text}")

    '''
    out_1 = predictor.predict_generate_beamsearch(text,
                                                  beam_size=1,
                                                  input_max_length=512,
                                                  out_max_length=100)
    print(f"out_1 is {out_1}")
    '''

    texts = [
        #"1月7日，五华区召开“中共昆明市五华区委十届三次全体(扩大)会议”，",
        #"1月7日，五华区召开“中共昆明市五华区委十届三次全体(扩大)会议”，区委书记金幼和作了《深入学习贯彻党的十八大精神，奋力开创五华跨越发展新局面》的工作报告。",
        #"拥有美丽身材是大多数女人追求的梦想，甚至有不少mm为了实现这个梦而心甘情愿付出各种代价，",
        #"2007年乔布斯向人们展示iPhone并宣称它将会改变世界",
        #"从前有座山，",
        #"如何摆脱无效焦虑?",
        #"许嵩是",
        #"北京在哪儿?",
        #"北京",
        #"where is Beijing?",
        #"汽车EDR是什么",
        #"今天天气不错，",
        "My favorite animal is",
        "今天天气不错",
        #"如何评价许嵩?",
        #"汽车EDR是什么",
        #"给妈妈送生日礼物，怎么选好？",
        #"1加1等于18497是正确的吗？",
        #"如何给汽车换胎？",
        #"以初春、黄山为题，做一首诗。",
        #"What is machine learning?",
        #"Machine learning is",
        #"Nigerian billionaire Aliko Dangote says he is planning a bid to buy the UK Premier League football club.",
        #"Hollym Gate railway station",
        #"Q:\n\nWhat is is machine learning?",
        "北京市统计局今天发布的“国际一流的和谐宜居之都”建设监测评价结果显示，",
        "Beijing is located",
        "NBA is"
    ]

    import random
    import numpy as np
    import torch
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for text in texts:
        print(f"inp is {text}")
        for itr in range(0, 3):
            out_2 = predictor.predict_generate_randomsample(text,
                                                            input_max_length=512,
                                                            out_max_length=100,
                                                            repetition_penalty=2.5,
                                                            top_k=50,
                                                            top_p=0.9)
        
            print(f"out_2 is {out_2}")

            '''
            out_1 = predictor.predict_generate_beamsearch(text,
                                                          beam_size=1,
                                                          input_max_length=512,
                                                          out_max_length=100)
            print(f"out_1 is {out_1}")
            '''

