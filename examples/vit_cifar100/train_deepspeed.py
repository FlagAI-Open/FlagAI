import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader

lr = 2e-5
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_type = "deepspeed"
trainer = Trainer(
    env_type=env_type,
    experiment_name="vit-cifar100-deepspeed",
    batch_size=150,
    num_gpus=8,
    fp16=True,
    gradient_accumulation_steps=1,
    lr=lr,
    weight_decay=1e-5,
    epochs=n_epochs,
    log_interval=100,
    eval_interval=1000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_vit_cifar100_deepspeed",
    save_interval=1000,
    num_checkpoints=1,
    hostfile="./hostfile",
    deepspeed_config='./deepspeed.json',
    training_script="train_deepspeed.py"
)

def build_cifar():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR100(root="./data/cifar100", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root="./data/cifar100", train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    if trainer.fp16:
        images = images.half()
    labels = [b[1] for b in batch]
    labels = torch.tensor(labels).long()
    return {"images": images, "labels": labels}

def validate(logits, labels, meta=None):
    _, predicted = logits.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    return correct / total

if __name__ == '__main__':
    loader = AutoLoader(task_name="classification",
                        model_name="vit-base-p16-224",
                        num_classes=100)

    model = loader.get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_dataset, val_dataset = build_cifar()

    trainer.train(model,
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  train_dataset=train_dataset,
                  valid_dataset=val_dataset,
                  metric_methods=[["accuracy", validate]],
                  collate_fn=collate_fn)





