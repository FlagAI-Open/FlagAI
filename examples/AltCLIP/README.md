



# AltCLIP

## 简介/Overview

我们提出了一个简单高效的方法去训练更加优秀的双语CLIP模型。命名为AltCLIP。AltCLIP基于 [Stable Diffusiosn](https://github.com/CompVis/stable-diffusion) 训练，训练数据来自 [WuDao数据集](https://data.baai.ac.cn/details/WuDaoCorporaText) 和 [LIAON](https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus) 

AltCLIP模型可以为本项目中的AltDiffusion模型提供支持，关于AltDiffusion模型的具体信息可查看[此教程](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion/README.md) 。

模型代码已经在 [FlagAI](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) 上开源，权重位于我们搭建的 [modelhub](https://model.baai.ac.cn/model-detail/100075) 上。我们还提供了微调，推理，验证的脚本，欢迎试用。

首次运行AltCLIP时，下列权重将会自动从modelhub上下载。

| 模型名称 Model name | 大小 Size | 描述 Description                                   |
| ------------------- | --------- | -------------------------------------------------- |
| AltCLIP             | 3.22G     | 我们的双语AltCLIP模型；Our bilingual AltCLIP model |



We propose a simple and efficient method to train a better bilingual CLIP model. Named AltCLIP. AltCLIP is trained based on [Stable Diffusiosn](https://github.com/CompVis/stable-diffusion) with training data from [WuDao dataset](https://data.baai.ac.cn/details/WuDaoCorporaText) and [Liaon](https://huggingface.co/datasets/laion/laion2B-en).

The AltCLIP model can provide support for the AltDiffusion model in this project. Specific information on the AltDiffusion model can be found in [this tutorial](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion/README.md).

The model code has been open sourced on [FlagAI](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) and the weights are located on [modelhub](https://model.baai.ac.cn/model-detail/100075). We also provide scripts for fine-tuning, inference, and validation, so feel free to try them out.

## 引用
关于AltCLIP，我们已经推出了相关论文，有更多细节可以查阅，如对您的工作有帮助，欢迎引用。

If you find this work helpful, please consider to cite
```
@article{https://doi.org/10.48550/arxiv.2211.06679,
  doi = {10.48550/ARXIV.2211.06679},
  url = {https://arxiv.org/abs/2211.06679},
  author = {Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences},
  title = {AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


## 训练/Training

训练共有两个阶段。
在平行知识蒸馏阶段，我们只是使用平行语料文本来进行蒸馏（平行语料相对于图文对更容易获取且数量更大）。在双语对比学习阶段，我们使用少量的中-英 图像-文本对（一共约2百万）来训练我们的文本编码器以更好地适应图像编码器。

There are two phases of training.
In the parallel knowledge distillation phase, we only use parallel corpus texts for distillation (parallel corpus is easier to obtain and larger in number compared to image text pairs). In the bilingual comparison learning phase, we use a small number of Chinese-English image-text pairs (about 2 million in total) to train our text encoder to better fit the image encoder.



## 下游效果/Performance

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
      <td>AltCLIP</td>
      <td>66.3 </td>
      <td>87.8 </td>
      <td>92.7 </td>
      <td>85.9 </td>
      <td>97.7 </td>
      <td>99.1 </td>
      <td>88.3 </td>
   </tr>
   <tr>
      <td>AltCLIP∗</td>
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
      <td>AltCLIP</td>
      <td>63.7 </td>
      <td>86.3 </td>
      <td>92.1 </td>
      <td>84.7 </td>
      <td>97.4 </td>
      <td>98.7 </td>
      <td>87.2 </td>
   </tr>
   <tr>
      <td>AltCLIP∗</td>
      <td>69.8 </td>
      <td>89.9 </td>
      <td>94.7 </td>
      <td>84.8 </td>
      <td>97.4 </td>
      <td>98.8 </td>
      <td>89.2 </td>
   </tr>
</table>

![image-20221111172255521](https://raw.githubusercontent.com/920232796/test/master/image.png)




## 可视化效果/Visualization effects

基于AltCLIP，我们还开发了AltDiffusion模型，可视化效果如下。

Based on AltCLIP, we have also developed the AltDiffusion model, visualized as follows.

![](https://raw.githubusercontent.com/920232796/test/master/image7.png)

## 模型推理 Inference

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
    model_name="AltCLIP-XLMR-L"
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

## CLIP微调/Finetuning

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
    model_dir="./checkpoints/",
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
    input_ids = torch.tensor([tokenizer(f"a photo of a {b[1]}",padding=True,truncation=True,max_length=77)["input_ids"] for b in batch])    

    return {
        "pixel_values": images,
        "input_ids": input_ids
    }
    
if __name__ == "__main__":
    trainer.train(model=model, train_dataset=dataset, collate_fn=cifar10_collate_fn)
```



## 模型验证/Evaluation

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
    model_name="AltCLIP-XLMR-L"
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
        classnames = dataset.classes if hasattr(dataset, "classes") else None

        zeroshot_templates = template["cifar10"]
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

```

