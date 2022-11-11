# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.auto_model.auto_loader import AutoLoader
import zeroshot_classification
import json 
import os 
from torchvision.datasets import CIFAR10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxlen = 256

dataset_root = "clip_benchmark_datasets"
dataset_name = "cifar10"

auto_loader = AutoLoader(
    task_name="txt_img_matching",
    model_dir="./checkpoints/",
    model_name="AltCLIP-XLMR-L"   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
)

model = auto_loader.get_model()
model.to(device)
model.eval()
tokenizer = auto_loader.get_tokenizer()
transform = auto_loader.get_transform()

dataset = CIFAR10(root=os.path.join(dataset_root, dataset_name), 
                transform=transform,   
                download=True)
batch_size = 128
num_workers = 4

template = {"cifar10": [
        "a photo of a {c}.",
        "a blurry photo of a {c}.",
        "a black and white photo of a {c}.",
        "a low contrast photo of a {c}.",
        "a high contrast photo of a {c}.",
        "a bad photo of a {c}.",
        "a good photo of a {c}.",
        "a photo of a small {c}.",
        "a photo of a big {c}.",
        "a photo of the {c}.",
        "a blurry photo of the {c}.",
        "a black and white photo of the {c}.",
        "a low contrast photo of the {c}.",
        "a high contrast photo of the {c}.",
        "a bad photo of the {c}.",
        "a good photo of the {c}.",
        "a photo of the small {c}.",
        "a photo of the big {c}."
    ],
}
def evaluate():
    if dataset:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        zeroshot_templates = template["cifar10"]
        classnames = dataset.classes if hasattr(dataset, "classes") else None

        metrics = zeroshot_classification.evaluate(
            model,
            dataloader,
            tokenizer,
            classnames, 
            zeroshot_templates,
            device=device,
            amp=True,
        )
       
        dump = {
            "dataset": dataset_name,
            "metrics": metrics
        }

        print(dump)
        with open("./result.txt", "w") as f:
            json.dump(dump, f)
        return metrics

if __name__ == "__main__":
    evaluate()
