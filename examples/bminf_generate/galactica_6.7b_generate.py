
from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader
import torch
import bminf
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


loader = AutoLoader(task_name="lm",
                    model_name="galactica-6.7b-en",
                    model_dir="./checkpoints/")

model = loader.get_model()
with torch.cuda.device(0):
    model = bminf.wrapper(model, quantization=False, memory_limit=20 << 30)
model.to(device)
model.eval()
tokenizer = loader.get_tokenizer()
predictor = Predictor(model, tokenizer)
print("model loaded")
time_start=time.time()

text = "Please write a abstract about the computer vision. \n"
out = predictor.predict_generate_randomsample(text,
                                            out_max_length=700,
                                            top_k=50,
                                            repetition_penalty=1.2,
                                            temperature=0.7
                                            )

time_end=time.time()
print('time cost',time_end-time_start,'s')
print(out)



