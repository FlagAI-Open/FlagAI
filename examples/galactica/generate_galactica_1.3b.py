from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader
import torch
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="lm",
                    model_name="galactica-1.3b-en",
                    model_dir="/share/projset/baaishare/baai-mrnd/xingzhaohu/")

model = loader.get_model()
model.to(device)
model.eval()

tokenizer = loader.get_tokenizer()

predictor = Predictor(model, tokenizer)

text = "Please write a abstract about the computer vision. \n"
out = predictor.predict_generate_randomsample(text,
                                              out_max_length=700,
                                              top_k=50,
                                              repetition_penalty=1.2,
                                              temperature=0.7
                                              )
print(out)