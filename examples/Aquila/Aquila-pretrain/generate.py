import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.model.predictor.predictor import Predictor
import bminf

state_dict = "./checkpoints_in/"
model_name = 'aquila-7b'

loader = AutoLoader("lm",
                    model_dir=state_dict,
                    model_name=model_name,
                    use_cache=True,
                    device='cuda',
                    fp16=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()

model.eval()
model.cuda()

texts = [
    "汽车EDR是什么",
]

predictor = Predictor(model, tokenizer)

for text in texts:
    print('-' * 80)
    text = f'{text}'
    print(f"text is {text}")
    with torch.no_grad():
        out = predictor.predict_generate_randomsample(text,
                                                      out_max_length=200,
                                                      top_p=0.95)
        print(f"pred is {out}")
