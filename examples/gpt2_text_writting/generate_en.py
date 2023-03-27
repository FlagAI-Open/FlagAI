# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import bminf
import time

if __name__ == '__main__':
    
    loader = AutoLoader("seq2seq",
                        "GPT2-base-en",
                        model_dir="./checkpoints/")
    model = loader.get_model()
    model = model.to('cpu')
    tokenizer = loader.get_tokenizer()
    time_start=time.time()
    with torch.cuda.device(0):
        model = bminf.wrapper(model, quantization=False, memory_limit=20 << 30)
    predictor = Predictor(model, tokenizer)

    text = "What is machine learning"

    out_2 = predictor.predict_generate_randomsample(text,
                                                    input_max_length=512,
                                                    out_max_length=100,
                                                    repetition_penalty=1.5,
                                                    top_k=20,
                                                    top_p=0.8)
            
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    # print(f"out_1 is {out_1}")
    print(f"out_2 is {out_2}")