from modelscope import AltDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = AltDiffusionPipeline.from_pretrained("BAAI/DreamBooth-AltDiffusion")
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "一张<鸣人>男孩的照片"
image = pipe(prompt, num_inference_steps=25).images[0]
image.show()