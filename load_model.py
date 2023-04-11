import sys;sys.path.append("/home/yanzhaodong/FlagAI")
import torch
from flagai.model.mm.AltDiffusion2 import LatentDiffusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LatentDiffusion.from_pretrain(download_path="./checkpoints", model_name="AltDiffusion-m18",device=device)
import pdb;pdb.set_trace()