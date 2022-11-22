import torch

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

loader = AutoLoader(task_name="lm",
                    model_name="opt-30b-en")

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()

text = "The trophy doesn’t fit in the suitcase because "
predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text,
                                              input_max_length=100,
                                              out_max_length=300,
                                              top_k=30,
                                              top_p=0.9,
                                              repetition_penalty=3.0)

print(f"input is {text} \n out is {out}")
