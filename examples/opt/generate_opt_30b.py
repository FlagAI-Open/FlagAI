from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader
import torch
import os
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# loader = AutoLoader(task_name="lm",
#                     model_name="opt-30b")
# loader.load_pretrain_params("/mnt/models_xingzhaohu/opt_30b")


from flagai.model.opt_model import OPTModel
from flagai.data.tokenizer.opt.opt_en_tokenizer import OPTTokenizer
print(f"正在构建模型")
model = OPTModel.init_from_json(os.path.join("/mnt/models_xingzhaohu/opt_30b", "config.json"))
tokenizer = OPTTokenizer()
model.load_weights("/mnt/models_xingzhaohu/opt_30b/pytorch_model.bin")



# model = loader.get_model()
# tokenizer = loader.get_tokenizer()
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