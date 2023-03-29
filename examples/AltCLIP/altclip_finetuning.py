# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.auto_model.auto_loader import AutoLoader
import os 
from flagai.trainer import Trainer
from torchvision.datasets import (
    CIFAR10
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = "./clip_benchmark_datasets"
dataset_name = "cifar10"

batch_size = 4
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

auto_loader = AutoLoader(
    task_name="txt_img_matching",
    model_dir="./checkpoints",
    model_name="AltCLIP-XLMR-L"   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
)

model = auto_loader.get_model()
model.to(device)
model.eval()
tokenizer = auto_loader.get_tokenizer()
transform = auto_loader.get_transform()

trainer = Trainer(env_type="pytorch",
                pytorch_device=device,
                experiment_name="clip_finetuning",
                batch_size=4,
                lr=1e-4,
                epochs=10,
                log_interval=10)

dataset = CIFAR10(root=os.path.join(dataset_root, dataset_name), 
                transform=transform,   
                download=True)

def cifar10_collate_fn(batch):
    # image shape is (batch, 3, 224, 224)
    images = torch.tensor([b[0]["pixel_values"][0] for b in batch])
    # text_id shape is (batch, n)
    input_ids = torch.tensor([tokenizer(f"a photo of a {b[1]}",
                                padding=True,
                                truncation=True,
                                max_length=77)["input_ids"] for b in batch])    

    attention_mask = torch.tensor([tokenizer(f"a photo of a {b[1]}",
                                padding=True,
                                truncation=True,
                                max_length=77)["attention_mask"] for b in batch])

    return {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
if __name__ == "__main__":
    trainer.train(model=model, train_dataset=dataset, collate_fn=cifar10_collate_fn)