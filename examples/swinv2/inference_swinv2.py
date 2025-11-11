import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
from flagai.auto_model.auto_loader import AutoLoader

data_path = "/data2/yzd/FlagAI/examples/swinv2/imagenet2012/"

# swinv2 model_name support:
# 1. swinv2-base-patch4-window16-256,
# 2. swinv2-small-patch4-window16-256,
# 3. swinv2-base-patch4-window8-256
loader = AutoLoader(task_name="classification",
                    model_name="swinv2-small-patch4-window16-256")
model = loader.get_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)

# imagenet loader
def data_loader(root, batch_size=256, workers=1):
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((256, 256)),
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

@torch.no_grad()
def test(model,data_loader):
    model.eval()
    top1_acc = 0.0
    top5_acc = 0.0

    for step, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)["logits"]
        _, top1_preds = outputs.max(1)
        top1_acc += top1_preds.eq(labels).sum().item()

        top5_pred = outputs.topk(5, 1, True)[1]
        top5_acc += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred).to(device)).sum().item()

    print(
        "test_top1_acc [{top1_acc}], test_top5_acc [{top5_acc}] \n".format(
            top1_acc=top1_acc/len(data_loader.dataset),
            top5_acc=top5_acc/len(data_loader.dataset),
        )
    )
if __name__ == '__main__':

    val_loader = data_loader(data_path, batch_size=8, workers=8)
    test(model, val_loader)
