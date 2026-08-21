# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader
import torch

loader = AutoLoader(task_name="lm",
                    model_name="opt-66b-en")

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()

text = """I think The Old Man and the Sea is a very good book, what do you think? Thank you for your question, I think """

predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text,
                                              input_max_length=100,
                                              out_max_length=300,
                                              top_k=50,
                                              top_p=0.9,
                                              repetition_penalty=3.0)

print(f"input is {text} \n out is {out}")