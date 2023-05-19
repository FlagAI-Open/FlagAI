import torch
from torch.utils.tensorboard import SummaryWriter

for ckpt in [178000, 180000]:
    print(f"add histogram:{ckpt}......")
    writer = SummaryWriter("/share/ldwang/tboard")
    a = torch.load(f"./{ckpt}/pytorch_model.bin", map_location='cpu')
    for key in a.keys():
        writer.add_histogram(f"ckpt-{key}/{key}", a[key], ckpt)
