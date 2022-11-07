import sys
sys.path.append("/home/yanzhaodong/FlagAI-internal/")
import torch
from flagai.auto_model.auto_loader import AutoLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = AutoLoader(task_name="txt_img_matching", #contrastive learning
                    model_name="clip-cn-b-16",
                    model_dir="/sharefs/baai-mrnd/yzd/")
model = loader.get_model()
model.to(device)
print(model.encode("两只老虎"))
