# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

if __name__ == '__main__':
    loader = AutoLoader("seq2seq",
                        "GPT2-base-ch",
                        model_dir="./checkpoints/")
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    predictor = Predictor(model, tokenizer)

    text = "今天天气不错"

    out_1 = predictor.predict_generate_beamsearch(text,
                                                  beam_size=5,
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
