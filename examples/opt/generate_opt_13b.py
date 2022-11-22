import torch

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="lm",
                    model_name="opt-13b-en")

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()
model.half()
model.to(device)

text = """How about the book The Old Man and the Sea?
        Thanks for your question, let me share my thoughts:"""

predictor = Predictor(model, tokenizer)
out = predictor.predict_generate_randomsample(text,
                                              input_max_length=100,
                                              out_max_length=300,
                                              top_k=30,
                                              top_p=0.9,
                                              repetition_penalty=3.0)

print(f"input is {text} \n out is {out}")
