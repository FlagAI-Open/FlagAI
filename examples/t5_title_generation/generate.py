# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import sys 
sys.path.append("/home/yanzhaodong/anhforth/FlagAI")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

if __name__ == '__main__':
    loader = AutoLoader("title-generation", "T5-base-ch", model_dir="./checkpoints")
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    predictor = Predictor(model, tokenizer)

    text = "一辆小轿车一名女司机竟造成9死24伤日前深圳市交警局对事故进行通报：" \
           "从目前证据看事故系司机超速行驶且操作不当导致目前24名伤员已有6名治愈出院其余正接受治疗预计事故赔偿费或超一千万元"

    out_1 = predictor.predict_generate_beamsearch(text,
                                                  beam_size=3,
                                                  input_max_length=512,
                                                  out_max_length=100)
    out_2 = predictor.predict_generate_randomsample(text,
                                                    input_max_length=512,
                                                    out_max_length=100,
                                                    repetition_penalty=1.5,
                                                    top_k=20,
                                                    top_p=0.8)

    print(f"out_1 is {out_1}")
    print(f"out_2 is {out_2}")
