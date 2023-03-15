# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

# Initialize 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="text2img", #contrastive learning
                    model_name="AltDiffusion",
                    model_dir="./checkpoints",
                    use_fp16=False)

model = loader.get_model()
model.eval()
model.to(device)
predictor = Predictor(model)
predictor.predict_generate_images(
    "Anime portrait of natalie portman as an anime girl by stanley artgerm lau, wlop, rossdraws, james jean, andrei riabovitchev, marc simonetti, and sakimichan, trending on artstation"
)
