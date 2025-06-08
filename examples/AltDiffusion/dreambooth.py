import os
import sys
import random
import itertools
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
from PIL import Image
from torchvision import transforms

from flagai.auto_model.auto_loader import AutoLoader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== 训练参数配置 ===============
instance_data_dir = "./instance_images"  # 实例图片目录
instance_prompt = "smile😁"          # 实例提示词（包含特殊标识符）

with_prior_preservation = False      # 是否使用先验保留
class_data_dir = "Mix"               # 类别图片目录
class_prompt = "smile"                 # 类别提示词
prior_loss_weight = 1.0              # 先验损失权重
num_class_images = 4                 # 类别图片数量
resolution = 128                     # 图片分辨率
center_crop = False                  # 是否中心裁剪

train_text_encoder = False           # 是否训练文本编码器
train_only_unet = True               # 是否仅训练UNet

# 训练超参数
num_train_epochs = 10                # 训练轮数
batch_size = 2                       # 批次大小
learning_rate = 5e-6                 # 学习率
adam_beta1 = 0.9                     # Adam优化器参数
adam_beta2 = 0.999                   # Adam优化器参数
adam_weight_decay = 1e-2             # 权重衰减
adam_epsilon = 1e-08                 # 数值稳定性常数

# =============== 模型初始化 ===============
# 加载AltDiffusion-m18文本生成图像模型
auto_loader = AutoLoader(task_name="text2img",
                         model_name="AltDiffusion-m18")

model = auto_loader.get_model()      # 获取模型
tokenizer = model.tokenizer          # 获取分词器

# =============== 数据集定义 ===============
class DreamBoothDataset(Dataset):
    def __init__(
            self,
            instance_data_root,    # 实例图片路径
            instance_prompt,       # 实例提示词
            tokenizer,             # 分词器
            class_data_root=None,  # 类别图片路径
            class_prompt=None,     # 类别提示词
            size=512,              # 图片大小
            center_crop=False,     # 是否中心裁剪
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # 处理实例图片
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        # 处理类别图片（用于先验保留）
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

        # 图片预处理流程
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 归一化到[-1, 1]
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # 加载实例图片
        path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        # 应用预处理并获取提示词token
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        example["caption"] = self.instance_prompt

        # 加载类别图片（如果使用先验保留）
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

# =============== 创建数据集和数据加载器 ===============
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
    # 批量处理数据
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    captions = [example["caption"] for example in examples]

    # 合并实例和类别数据（如果使用先验保留）
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
        "txt": captions
    }
    return batch

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

# =============== 模型组件准备 ===============
vae = model.first_stage_model         # 变分自编码器
text_encoder = model.cond_stage_model # 文本编码器
unet = model.model.diffusion_model    # 扩散模型（UNet）

# 冻结不需要训练的组件
vae.requires_grad_(False)
if not train_text_encoder:
    text_encoder.requires_grad_(False)

# =============== 优化器设置 ===============
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

# =============== 混合精度训练 ===============
scaler = GradScaler()  # 梯度缩放器（用于混合精度训练）

# 将模型移到设备
model.to(device)
vae = model.first_stage_model.to(device)
text_encoder = model.cond_stage_model.to(device)
unet = model.model.diffusion_model.to(device)

# 训练循环
for epoch in range(num_train_epochs):
    unet.train()
    if train_text_encoder:
        text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        # 获取输入数据
        x, c = model.get_input(batch, "pixel_values")

        # 混合精度训练
        with autocast():  # 自动混合精度上下文
            if with_prior_preservation:
                # 分离实例和先验数据
                x, x_prior = torch.chunk(x, 2, dim=0)
                c, c_prior = torch.chunk(c, 2, dim=0)
                # 计算实例损失和先验损失
                loss, _ = model(x, c)
                prior_loss, _ = model(x_prior, c_prior)
                # 组合损失
                loss = loss + prior_loss_weight * prior_loss
            else:
                # 仅计算实例损失
                loss, _ = model(x, c)

        print('*' * 20, "loss=", str(loss.detach().item()))

        # 反向传播和优化
        scaler.scale(loss).backward()  # 缩放梯度
        scaler.step(optimizer)        # 更新参数
        scaler.update()               # 更新缩放器
        optimizer.zero_grad()         # 清零梯度

# =============== 保存训练好的模型 ===============
checkpoint_path = './checkpoints/AltDiffusion-m18-new-trained/model.ckpt'
# 确保目录存在
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved to {checkpoint_path}")
