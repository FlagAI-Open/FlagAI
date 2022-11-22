import os

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from flagai.auto_model.auto_loader import AutoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./imagenet2012/"

loader = AutoLoader(task_name="classification",
                    model_name='swinv1-base-patch4-window7-224',
                    num_classes=1000)
model = loader.get_model()
model.eval()
model = model.to(device)

def data_loader(root, batch_size=64, workers=8):
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers
                            )

    return val_loader

# 测试预训练权重
@torch.no_grad()
def test(model,data_loader):

    model.eval()
    top1_acc = 0.0

    for step, (inputs, labels) in enumerate(data_loader):

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(images=inputs)["logits"]

        _, top1_preds = outputs.max(1)
        top1_acc += top1_preds.eq(labels).sum().item()


    print(
            "test_top1_acc [{top1_acc}] \n".format(
             top1_acc=top1_acc/len(data_loader.dataset),
            )
    )

if __name__ == '__main__':

    val_loader = data_loader(data_path)
    test(model,val_loader)
