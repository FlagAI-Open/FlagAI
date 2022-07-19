# Vit for classification with cifar100 dataset

Vision Transformer(Vit) is becoming increasingly popular in the field of 
compute vision(CV). More and more tasks are using Vit to achieve the SOTA.

The paper is in https://arxiv.org/pdf/2010.11929.pdf.

Code is in https://github.com/google-research/vision_transformer.

## How to use
We can easily use the Vit to finetune cifar100 dataset.
### Training
```python
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

trainer = Trainer(
   env_type="pytorch",
   experiment_name="vit-cifar100",
   batch_size=64,
   gradient_accumulation_steps=1,
   lr=lr,
   weight_decay=1e-5,
   epochs=n_epochs,
   log_interval=100,
   eval_interval=1000,
   load_dir=None,
   pytorch_device=device,
   save_dir="checkpoints_vit_cifar100",
   save_interval=1000,
   num_checkpoints=1,
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

   train_dataset = CIFAR100(root="./cifar100", train=True, download=True, transform=transform_train)
   test_dataset = CIFAR100(root="./cifar100", train=False, download=True, transform=transform_test)
   return train_dataset, test_dataset

def collate_fn(batch):
   images = torch.stack([b[0] for b in batch])
   labels = [b[1] for b in batch]
   labels = torch.tensor(labels).long()
   return {"images": images, "labels": labels}


def validate(logits, labels, meta=None):
   _, predicted = logits.max(1)
   total = labels.size(0)
   correct = predicted.eq(labels).sum().item()
   return correct / total


if __name__ == '__main__':
   loader = AutoLoader(task_name="backbone",
                       model_name="Vit-base-p16",
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
```

### Validation
If you have trained a model, you can valite it again by following code.
```python
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from flagai.auto_model.auto_loader import AutoLoader
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_cifar():

   transform_test = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])

   test_dataset = CIFAR100(root="./cifar100", train=False, download=True, transform=transform_test)
   return test_dataset

def collate_fn(batch):
   images = torch.stack([b[0] for b in batch])
   labels = [b[1] for b in batch]
   labels = torch.tensor(labels).long()
   return {"images": images, "labels": labels}

def validate(logits, labels, meta=None):
   _, predicted = logits.max(1)
   total = labels.size(0)
   correct = predicted.eq(labels).sum().item()
   return correct / total

if __name__ == '__main__':

   model_save_dir = "./checkpoints_vit_cifar100"
   print(f"loadding model in :{model_save_dir}")
   loader = AutoLoader(task_name="backbone",
                       model_name="Vit-base-p16",
                       num_classes=100)

   model = loader.get_model()

   model.load_state_dict(torch.load(os.path.join(model_save_dir, "38000", "pytorch_model.bin"), map_location=device)["module"])
   print(f"model load success.......")
   model.to(device)

   val_dataset = build_cifar()

   val_dataloader = DataLoader(val_dataset,
                               batch_size=1,
                               shuffle=False,
                               collate_fn=collate_fn)
   index = 0
   accuracy = 0.0
   for data in tqdm(val_dataloader, total=len(val_dataloader)):
      index += 1
      data = {k: v.to(device) for k, v in data.items()}
      labels = data["labels"]
      pred = model(**data)["logits"]
      acc = validate(pred, labels)
      accuracy += acc

   print(f"accuracy is {accuracy / index}")
```
