# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import sys;sys.path.append("/home/yanzhaodong/FlagAI")
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

# Initialize 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="text2img", #contrastive learning
                    model_name="AltDiffusion-m18",
                    model_dir="./checkpoints",
                    use_fp16=False,)
# loader = AutoLoader(task_name="text2img", #contrastive learning
#                     model_name="AltDiffusion-m18")
model = loader.get_model()
# for name, param in model.named_parameters():
#     if name.startswith("cond_stage_model"):
#         print(name)
# import pdb;pdb.set_trace()
for name, param in model.named_parameters():
    if 'pre_LN' in name:
        print(name,param)


model.eval()
model.to(device)
predictor = Predictor(model)
prompt = "Daenerys Targaryen as a mermeid with a piercing gaze wearing an enchanted bikini in an underwater magical forest, highly detailed face, realistic face, beautiful detailed eyes, fantasy art, in the style of artgerm, illustration, epic, fantasy, intricate, hyper detailed, artstation, concept art, smooth, sharp focus, ray tracing, vibrant, photorealistic"
negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, extra head, extra legs,fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
seed = 553124
W = 512 
H = 512
scale=7.5
predictor.predict_generate_images_m18(
    prompt=prompt,negative_prompt=negative_prompt,seed=seed, W=W, H=H, scale=scale
)
