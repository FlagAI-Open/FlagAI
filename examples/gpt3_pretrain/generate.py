# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

if __name__ == '__main__':
    loader = AutoLoader("seq2seq",
                        "gpt2-base-en",
                        model_dir="./")
    model = loader.get_model()
    model.cuda()
    tokenizer = loader.get_tokenizer()
    predictor = Predictor(model, tokenizer)

    text = "Hollym Gate railway station"
    text = "Molly Henderson Molly Henderson (born September 14, 1953) is a former Commissioner of Lancaster County, Pennsylvania. The Commissioners are"
    text = "Major League Baseball All-Century Team In 1999, the Major League Baseball"
    text = "Nigerian billionaire Aliko Dangote says he is planning a bid to buy the UK Premier League football club. "

    print(f"inp is {text}")

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
