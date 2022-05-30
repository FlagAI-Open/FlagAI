
from flagai.auto_model.auto_loader import AutoLoader

import torch

loader = AutoLoader(task_name="lm",
                   model_name="GLM-large-ch",
                   only_download_config=True)


model = loader.get_model()
for k, v in model.named_parameters():
    print(k)



# n1 = torch.nn.Linear(3, 4)
#
# n1.nam