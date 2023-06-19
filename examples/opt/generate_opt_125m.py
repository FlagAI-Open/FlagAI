# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="lm",
                    model_name="opt-125m-en")

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()
model.to(device)

text = "The trophy doesn’t fit in the suitcase because "
predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text,
                                              input_max_length=100,
                                              out_max_length=300,
                                              top_k=30,
                                              top_p=0.9,
                                              repetition_penalty=3.0)

print(f"input is {text} \n out is {out}")


