import os

import torch
from torchvision import datasets
from torchvision import transforms

from flagai.auto_model import AutoLoader
from flagai.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./imagenet2012/"

# use DDP for training by 4 gpus.
trainer = Trainer(env_type="pytorchDDP",
                  epochs=10,
                  experiment_name="swinv2_imagenet_ddp",
                  batch_size=32,
                  weight_decay=1e-3,
                  warm_up=0.1,
                  lr=5e-5,
                  save_interval=100,
                  eval_interval=100,
                  log_interval=10,
                  num_gpus=4,
                  hostfile="./hostfile",
                  training_script="training_swinv2.py"
                  )

# swinv2 model_name support:
# 1. swinv2-base-patch4-window16-256,
# 2. swinv2-small-patch4-window16-256,
# 3. swinv2-base-patch4-window8-256
loader = AutoLoader(task_name="classification",
                    model_name="swinv2-base-patch4-window8-256",
                    num_classes=1000)
model = loader.get_model()

# build imagenet dataset
def build_dataset(root):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize
        ])
    )
    return train_dataset, val_dataset

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    if trainer.fp16:
        images = images.half()
    labels = [b[1] for b in batch]
    labels = torch.tensor(labels).long()
    return {"images": images, "labels": labels}

def top1_acc(pred, labels, **kwargs):
    pred = pred.argmax(dim=1)
    top1_acc = pred.eq(labels).sum().item() / len(pred)
    return top1_acc

if __name__ == '__main__':

    print("building imagenet dataset......")
    train_dataset, val_dataset = build_dataset(root=data_path)
    print("training......")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    trainer.train(model,
                  train_dataset=train_dataset,
                  valid_dataset=val_dataset,
                  collate_fn=collate_fn,
                  optimizer=optimizer,
                  metric_methods=[["top1_acc", top1_acc]],
                  find_unused_parameters=False)
