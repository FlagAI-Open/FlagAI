

我们提出了一个简单高效的方法去训练更加优秀的双语CLIP模型。命名为AltCLIP。

We propose a simple and efficient method to train a better bilingual CLIP model. It is named AltCLIP.



训练共有两个阶段。
在平行知识蒸馏阶段，我们只是使用平行语料文本来进行蒸馏（平行语料相对于图文对更容易获取且数量更大）。在双语对比学习阶段，我们使用少量的中-英 图像-文本对（一共约2百万）来训练我们的文本编码器以更好地适应图像编码器。

There are two phases of training.
In the parallel knowledge distillation phase, we only use parallel corpus texts for distillation (parallel corpus is easier to obtain and larger in number compared to image text pairs). In the bilingual comparison learning phase, we use a small number of Chinese-English image-text pairs (about 2 million in total) to train our text encoder to better fit the image encoder.



模型与权重已经在FlagAI(https://github.com/FlagAI-Open/FlagAI)上开源，我们还提供了微调，推理，验证的脚本，欢迎试用。

The model and weights have been open sourced on FlagAI (https://github.com/FlagAI-Open/FlagAI), and we also provide scripts for fine-tuning, inference, and evaluation, so feel free to try them out.

# 下游效果 Performance

<table>
   <tr>
      <td rowspan=2>Language</td>
      <td rowspan=2>Method</td>
      <td colspan=3>Text-to-Image Retrival</td>
      <td colspan=3>Image-to-Text Retrival</td>
      <td rowspan=2>MR</td>
   </tr>
   <tr>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
   </tr>
   <tr>
      <td rowspan=7>English</td>
      <td>CLIP</td>
      <td>65.0 </td>
      <td>87.1 </td>
      <td>92.2 </td>
      <td>85.1 </td>
      <td>97.3 </td>
      <td>99.2 </td>
      <td>87.6 </td>
   </tr>
   <tr>
      <td>Taiyi</td>
      <td>25.3 </td>
      <td>48.2 </td>
      <td>59.2 </td>
      <td>39.3 </td>
      <td>68.1 </td>
      <td>79.6 </td>
      <td>53.3 </td>
   </tr>
   <tr>
      <td>Wukong</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>R2D2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>CN-CLIP</td>
      <td>49.5 </td>
      <td>76.9 </td>
      <td>83.8 </td>
      <td>66.5 </td>
      <td>91.2 </td>
      <td>96.0 </td>
      <td>77.3 </td>
   </tr>
   <tr>
      <td>AlterCLIP</td>
      <td>66.3 </td>
      <td>87.8 </td>
      <td>92.7 </td>
      <td>85.9 </td>
      <td>97.7 </td>
      <td>99.1 </td>
      <td>88.3 </td>
   </tr>
   <tr>
      <td>AlterCLIP∗</td>
      <td>72.5 </td>
      <td>91.6 </td>
      <td>95.4 </td>
      <td>86.0 </td>
      <td>98.0 </td>
      <td>99.1 </td>
      <td>90.4 </td>
   </tr>
   <tr>
      <td rowspan=7>Chinese</td>
      <td>CLIP</td>
      <td>0.0 </td>
      <td>2.4 </td>
      <td>4.0 </td>
      <td>2.3 </td>
      <td>8.1 </td>
      <td>12.6 </td>
      <td>5.0 </td>
   </tr>
   <tr>
      <td>Taiyi</td>
      <td>53.7 </td>
      <td>79.8 </td>
      <td>86.6 </td>
      <td>63.8 </td>
      <td>90.5 </td>
      <td>95.9 </td>
      <td>78.4 </td>
   </tr>
   <tr>
      <td>Wukong</td>
      <td>51.7 </td>
      <td>78.9 </td>
      <td>86.3 </td>
      <td>76.1 </td>
      <td>94.8 </td>
      <td>97.5 </td>
      <td>80.9 </td>
   </tr>
   <tr>
      <td>R2D2</td>
      <td>60.9 </td>
      <td>86.8 </td>
      <td>92.7 </td>
      <td>77.6 </td>
      <td>96.7 </td>
      <td>98.9 </td>
      <td>85.6 </td>
   </tr>
   <tr>
      <td>CN-CLIP</td>
      <td>68.0 </td>
      <td>89.7 </td>
      <td>94.4 </td>
      <td>80.2 </td>
      <td>96.6 </td>
      <td>98.2 </td>
      <td>87.9 </td>
   </tr>
   <tr>
      <td>AlterCLIP</td>
      <td>63.7 </td>
      <td>86.3 </td>
      <td>92.1 </td>
      <td>84.7 </td>
      <td>97.4 </td>
      <td>98.7 </td>
      <td>87.2 </td>
   </tr>
   <tr>
      <td>AlterCLIP∗</td>
      <td>69.8 </td>
      <td>89.9 </td>
      <td>94.7 </td>
      <td>84.8 </td>
      <td>97.4 </td>
      <td>98.8 </td>
      <td>89.2 </td>
   </tr>
</table>



![](https://raw.githubusercontent.com/920232796/test/master/image6.png)

# 可视化效果 Visualization effects

基于AltCLIP，我们还开发了AltDiffusion模型，可视化效果如下。

Based on AltCLIP, we have also developed the AltDiffusion model, visualized as follows.

![](https://raw.githubusercontent.com/920232796/test/master/image7.png)

# 模型推理 Inference

```python
import torch
from PIL import Image
from flagai.auto_model.auto_loader import AutoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## 一行代码直接自动下载权重到'./checkpoints/clip-xlmr-large'，并自动加载CLIP模型权重
## modelhub地址: Modelhub(https://model.baai.ac.cn/models)
loader = AutoLoader(
    task_name="txt_img_matching",
    model_dir="./checkpoints",
    model_name="clip-xlmr-large"
)
## 获取加载好的模型
model = loader.get_model()
## 获取tokenizer
tokenizer = loader.get_tokenizer()
## 获取transform用来处理图像
transform = loader.get_transform()

model.eval()
model.to(device)

## 推理过程,图像与文本匹配
image = Image.open("./dog.jpeg")
image = transform(image)
image = torch.tensor(image["pixel_values"]).to(device)
text = tokenizer(["a rat", "a dog", "a cat"])["input_ids"]

text = torch.tensor(text).to(device)

with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(text)
    text_probs = (image_features @ text_features.T).softmax(dim=-1)

print(text_probs.cpu().numpy()[0].tolist())
```

# CLIP微调 Finetuning

微调采用cifar10数据集，并使用FlagAI的Trainer快速开始训练过程。

Fine-tuning was done using the cifar10 dataset and using FlagAI's Trainer to quickly start the training process.

```python
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
    model_dir="/sharefs/baai-mrnd/xingzhaohu/",
    model_name="clip-xlmr-large"   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
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
    input_ids = torch.tensor([tokenizer(f"a photo of a {b[1]}",padding=True,truncation=True,max_length=77)["input_ids"] for b in batch])    

    return {
        "pixel_values": images,
        "input_ids": input_ids
    }
    
if __name__ == "__main__":
    trainer.train(model=model, train_dataset=dataset, collate_fn=cifar10_collate_fn)
```



# 模型验证 Evaluation

我们提供了可以直接运行的验证脚本，在cifar10数据集上进行验证。

期待的输出为：```{'dataset': 'cifar10', 'metrics': {'acc1': 0.95402, 'acc5': 0.99616, 'mean_per_class_recall': 0.9541200000000002}}```

We provide validation scripts that can be run directly on the cifar10 dataset.

```python
# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.auto_model.auto_loader import AutoLoader
from metrics import zeroshot_classification
import json 
import os 
from torchvision.datasets import CIFAR10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxlen = 256

dataset_root = "./clip_benchmark_datasets"
dataset_name = "cifar10"

auto_loader = AutoLoader(
    task_name="txt_img_matching",
    model_dir="./checkpoints/",
    model_name="clip-xlmr-large"
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
        metrics = zeroshot_classification.evaluate(
            model,
            dataloader,
            tokenizer,
            "cifar10", 
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

```

