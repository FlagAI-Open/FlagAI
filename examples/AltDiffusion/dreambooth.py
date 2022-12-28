# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# 
# https://huggingface.co/spaces/BAAI/dreambooth-altdiffusion/blob/main/train_dreambooth.py

import os
import sys
import random
import itertools
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

instance_data_dir = "./examples/AltDiffusion/instance_images"
instance_prompt = "<鸣人>男孩"

with_prior_preservation = False
class_data_dir = "Mix"
class_prompt = "男孩"
prior_loss_weight = 1.0
num_class_images = 10
resolution = 512
center_crop = True

train_text_encoder = False
train_only_unet = True

num_train_epochs = 500
batch_size = 4
learning_rate = 5e-6
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08

auto_loader = AutoLoader(task_name="text2img",
                         model_name="AltDiffusion")

model = auto_loader.get_model()
tokenizer = model.tokenizer

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            random.shuffle(self.class_images_path)
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        print('*'*20, "instance_prompt=", self.instance_prompt)
        example["caption"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

train_dataset = DreamBoothDataset(
    instance_data_root=instance_data_dir,
    instance_prompt=instance_prompt,
    class_data_root=class_data_dir if with_prior_preservation else None,
    class_prompt=class_prompt,
    tokenizer=tokenizer,
    size=resolution,
    center_crop=center_crop,
)

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    captions = [example["caption"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "caption": captions,
    }
    return batch

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

vae = model.first_stage_model
text_encoder = model.cond_stage_model
unet = model.model.diffusion_model

vae.requires_grad_(False)
if not train_text_encoder:
    text_encoder.requires_grad_(False)

optimizer_class = torch.optim.AdamW
params_to_optimize = (
    itertools.chain(unet.parameters(),
                    text_encoder.parameters()) if train_text_encoder else unet.parameters())
optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
)

model.to(device)
for epoch in range(num_train_epochs):
    unet.train()
    if train_text_encoder:
        text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        #x = batch["pixel_values"].to(device)
        #x = model.encode_first_stage(batch["pixel_values"]).to(device)
        #c = batch["caption"]
        x, c = model.get_input(batch, "pixel_values")

        if with_prior_preservation:
            x, x_prior = torch.chunk(x, 2, dim=0)
            c, c_prior = torch.chunk(c, 2, dim=0)
            loss, _ = model(x, c)
            prior_loss, _ = model(x_prior, c_prior)
            loss = loss + prior_loss_weight * prior_loss
        else:
            loss, _ = model(x, c)

        print('*'*20, "loss=", str(loss.detach().item()))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

## mkdir ./checkpoints/DreamBooth and copy ./checkpoints/AltDiffusion to ./checkpoints/DreamBooth/AltDiffusion
## overwrite model.ckpt for latter usage
chekpoint_path = './checkpoints/DreamBooth/AltDiffusion/model.ckpt'
torch.save(model.state_dict(), chekpoint_path)

