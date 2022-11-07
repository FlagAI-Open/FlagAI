from flagai.data.tokenizer.cpm_3.cpm3_tokenizer import CPM3Tokenizer
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import torch
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = AutoLoader("lm", "cpm3", model_dir="/sharefs/baai-mrnd/xw/")
model = loader.get_model()
tokenizer = CPM3Tokenizer('/sharefs/baai-mrnd/xw/cpm3/vocab.txt', space_token = '</_>', line_token = '</n>',)
predictor = Predictor(model, tokenizer)
model.to(device)
text = { "mode": "lm", "source": ["半坡饰族斜挎包牛皮女包包女宽肩带撞色条纹小方包运动风单肩包红色，条纹风格逆袭，柔软牛皮，\
    实用型收纳，时毗宽肩带，匠心工艺，深蓝，牛皮拉牌，大红，触感柔软，不同风格搭配的冲突感正是时毗的秘诀所在，于休闲中诠释自我。轻奢光泽坚韧耐用，1拉链暗袋，让代表性条纹装饰成为手袋的主角，实力与颜值集于一身。伴随着运动成为时尚生活方式，荔枝"],\
         "targets": "", "control": { "keywords": [], "genre": "", "relations": [], "events": [] } }
out_1 = predictor.predict_generate_beamsearch(text,
                                            beam_size=3,
                                            input_max_length=512,
                                            out_max_length=100)
print(out_1)