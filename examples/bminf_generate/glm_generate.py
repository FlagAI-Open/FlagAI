from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import Tokenizer
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import torch
import bminf

model_name = 'GLM-10b-ch'
loader = AutoLoader("lm", 'GLM-10b-ch', model_dir="./checkpoints/")
model = loader.get_model()
tokenizer = loader.get_tokenizer()
with torch.cuda.device(0):
    model = bminf.wrapper(model, quantization=False, memory_limit=30 << 39)

tokenizer = Tokenizer.from_pretrained(model_name)
predictor = Predictor(model, tokenizer)

text = "今天天气不错[gMASK]"
output = predictor.predict_generate_randomsample(text, out_max_length=10)
print(text, '\n', output)