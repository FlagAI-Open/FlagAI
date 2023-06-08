import os
import torch
import sys;sys.path.append("/data2/yzd/workspace/FlagAI")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.data.tokenizer import Tokenizer
import bminf

state_dict = "./checkpoints_in/"
model_name = 'aquila-7b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True)
model = loader.get_model()
# import pdb;pdb.set_trace()
tokenizer = loader.get_tokenizer()
# from flagai.model.aquila_model import AQUILAModel
# model = AQUILAModel.from
# tokenizer = Tokenizer.from_pretrained('aquila-7b', cache_dir='./checkpoints_in/aquila-7b')
pl_sd = torch.load('./checkpoints_in/aquila-7b/pytorch_model.bin', map_location="cpu")
if "state_dict" in pl_sd:
    sd = pl_sd["state_dict"]
else:
    sd = pl_sd
model.load_state_dict(sd, strict=True)

model.eval()

model.eval()
model.half()
# with torch.cuda.device(0):
#     model = bminf.wrapper(model, quantization=False, memory_limit=2 << 30)

model.cuda()


predictor = Predictor(model, tokenizer)

texts = [
        # "I am ",
        #"1月7日，五华区召开“中共昆明市五华区委十届三次全体(扩大)会议”，",
        #"1月7日，五华区召开“中共昆明市五华区委十届三次全体(扩大)会议”，区委书记金幼和作了《深入学习贯彻党的十八大精神，奋力开创五华跨越发展新局面》的工作报告。",
        #"拥有美丽身材是大多数女人追求的梦想，甚至有不少mm为了实现这个梦而心甘情愿付出各种代价，",
        #"2007年乔布斯向人们展示iPhone并宣称它将会改变世界",
        #"从前有座山，",
        #"如何摆脱无效焦虑?",
        "北京在哪儿?",
        #"北京",
        #"汽车EDR是什么",
        #"My favorite animal is",
        #"今天天气不错",
        #"如何评价许嵩?",
        #"汽车EDR是什么",
        #"给妈妈送生日礼物，怎么选好？",
        #"1加1等于18497是正确的吗？",
        #"如何给汽车换胎？",
        #"以初春、黄山为题，做一首诗。",
        #"What is machine learning?",
        #"Machine learning is",
        #"Nigerian billionaire Aliko Dangote says he is planning a bid to buy the UK Premier League football club.",
        #"The capital of Germany is the city of ",
        ]


for text in texts:
    print('-'*80)
    text = f'{text}' #base
    print(f"text is {text}")
    #out = predictor.predict_generate_randomsample(text, out_max_length=200,top_p=0.95)
    with torch.no_grad():
        # out = predictor.predict_generate_randomsample(text, out_max_length=200, temperature=0)
        out = predictor.predict_generate_randomsample(text, out_max_length=200,top_p=0.95)
        print(f"pred is {out}")
