# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    loader = AutoLoader("seq2seq",
                        "GPT2-base-ch",
                        model_dir="./state_dict/")
    model = loader.get_model()
    model.to(device)
    tokenizer = loader.get_tokenizer()
    predictor = Predictor(model, tokenizer)

    text = "今天天气不错"

    output = predictor.predict_generate_randomsample(text, input_max_length=512, out_max_length=100,
                                                        repetition_penalty=1.5, top_k=20, top_p=0.8)

    print(f"out is {output}")
