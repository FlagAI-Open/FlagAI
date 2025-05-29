import torch

ckpt = torch.load("/cdgm0705/hyx/train/output_qlora_calora/checkpoint-2000/adapter_model/lora.pt")

new_ckpt = {k: v for k, v in ckpt.items() if 'lora' in k}

print(new_ckpt.keys())

torch.save(new_ckpt, "/cdgm0705/hyx/train/output_qlora_calora/checkpoint-2000/adapter_model/lora-thin-2000.pt")
