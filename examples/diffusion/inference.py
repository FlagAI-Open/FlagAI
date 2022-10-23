import torch
from PIL import Image
import sys
sys.path.append("/home/yanzhaodong/FlagAI-internal/")
sys.path.append("/home/yanzhaodong/FlagAI-internal/flagai/model/mm/")
# sys.path.append("/home/yanzhaodong/FlagAI-internal/flagai/model/predictor/")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.dataset.mm.clip_dataset import clip_transform
from flagai.model.predictor.predictor import Predictor
import pdb

# Initialize 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader = AutoLoader(task_name="text2img", #contrastive learning
                    model_name="diffusion-ddpm-cnclip",
                    model_dir="/sharefs/baai-mrnd/yzd")

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()
model.to(device)

predictor = Predictor(model, tokenizer)
predictor.predict_generate_images("两只老虎")