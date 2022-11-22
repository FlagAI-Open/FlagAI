# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "./state_dict"

# Note "./checkpoints_seq2seq/{}/mp_rank_00_model_states.pt", {} is a directory in the checkpoints_seq2seq.
model_save_path = "./checkpoints_seq2seq/10/mp_rank_00_model_states.pt"
maxlen = 256
auto_loader = AutoLoader(
    "title-generation",
    model_name="RoBERTa-base-ch",
    model_dir=model_dir,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
predictor = Predictor(model, tokenizer)
model.load_state_dict(
    torch.load(model_save_path, map_location=device)["module"])
model.to(device)
model.eval()
test_data = [
    "本文总结了十个可穿戴产品的设计原则而这些原则同样也是笔者认为是这个行业最吸引人的地方1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人",
    "2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其c词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献",
    "雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"
]
for text in test_data:
    print(
        predictor.predict_generate_beamsearch(text,
                                              out_max_length=50,
                                              beam_size=3))
    print(
        predictor.predict_generate_randomsample(text,
                                                input_max_length=256,
                                                out_max_length=50,
                                                repetition_penalty=1.5,
                                                top_k=20,
                                                top_p=1.0,
                                                temperature=1.2))
