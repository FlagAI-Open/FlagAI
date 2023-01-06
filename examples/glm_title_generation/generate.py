# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_save_path = "./checkpoints/40/pytorch_model.bin"

auto_loader = AutoLoader("seq2seq",
                         model_name="GLM-large-ch",
                         model_dir="./state_dict/")
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

model.load_state_dict(
    torch.load(model_save_path, map_location=device)["module"])
    
model.to(device)
model.eval()
predictor = Predictor(model, tokenizer)

test_data = [
    "本文总结了十个可穿戴产品的设计原则，而这些原则同样也是笔者认为是这个行业最吸引人的地方，1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人。",
    "2007年乔布斯向人们展示iPhone并宣称它将会改变世界，还有人认为他在夸大其词，然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落，未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献。",
    "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东，截止发稿前雅虎股价上涨了大约7％至5145美元"
]

for text in test_data:
    print('===============================================\n')
    print(text, ":")
    for i in range(1):  #generate several times
        print("--------------sample %d :-------------------" % (i))
        print('-----------random sample: --------------')
        print(
            predictor.predict_generate_randomsample(text,
                                                    out_max_length=66,
                                                    top_k=10,
                                                    top_p=1.0,
                                                    repetition_penalty=4.0,
                                                    temperature=1.2))
        print('-----------beam search: --------------')
        print(
            predictor.predict_generate_beamsearch(text,
                                                  out_max_length=66,
                                                  beam_size=10))
