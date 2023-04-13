
# AltCLIP-m18


|      名称 Name       |  任务 Task   |   语言 Language(s)    | 模型 Model | Github |
|:------------------:|:----------:|:-------------------:|:--------:|:------:|
| AltCLIP-m18 |  Text-Image | Multilingual |   CLIP   |   [FlagAI](https://github.com/FlagAI-Open/FlagAI)   |

## 简介 Brief Introduction

继双语模型AltCLIP与9语模型AltCLIP-m9之后，我们训练了18语CLIP模型。命名为AltCLIP-m18。它支持英语、中文、日语、泰语、韩语、印地语、乌克兰语、阿拉伯语、土耳其语、越南语、波兰语、荷兰语、葡萄牙语、意大利语、西班牙语、德语、法语和俄语。

AltCLIP-m18模型可以为AltDiffusion-m18模型提供支持，关于AltDiffusion模型的具体信息可查看[此教程](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion/README.md) 。

模型代码已经在 [FlagAI](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP-m18) 上开源，权重位于我们搭建的 [modelhub](https://model.baai.ac.cn/model-detail/100095) 上。我们还提供了微调，推理，验证的脚本，欢迎试用。

Following the bilingual model AltCLIP and the nine-language model AltCLIP-m9, we trained the eighteen-language CLIP model, Named AltCLIP-m18. It supports English, Chinese, Japanese, Thai, Korean, Hindi, Ukrainian, Arabic, Turkish, Vietnamese, Polish, Dutch, Portuguese, Italian, Spanish, German, French, and Russian.

The AltCLIP-m18 model can provide support for the AltDiffusion-m18 model. Specific information on the AltDiffusion modle can be found in [this tutorial](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion/README.md).

The model code has been open sourced on [FlagAI](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP-m18) and the weights are located on [modelhub](https://model.baai.ac.cn/model-detail/100095). We also provide scripts for fine-tuning, inference, and validation, so feel free to try them out.

## 训练数据集 Training datasets



| No   | Language |                    Stage1(LAION400M)(MIT)                    |      |                         Stage 2 & 3                          |
| ---- | :------: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: |
| 1    |  **En**  |                                                              |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 2    |  **th**  |       [CCAligned](https://opus.nlpl.eu/CCAligned.php)        |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 3    |  **ko**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 4    |  **hi**  |       [CCAligned](https://opus.nlpl.eu/CCAligned.php)        |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 5    |  **uk**  |        [CCMatrix](https://opus.nlpl.eu/CCMatrix.php)         |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 6    |  **ar**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles-v2018.php) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 7    |  **tr**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), [CCMatrix](https://opus.nlpl.eu/CCMatrix.php) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 8    |  **vi**  |        [CCMatrix](https://opus.nlpl.eu/CCMatrix.php)         |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 9    |  **pl**  | [CCMatrix](https://opus.nlpl.eu/CCMatrix.php) , WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 10   |  **nl**  |        [CCMatrix](https://opus.nlpl.eu/CCMatrix.php)         |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 11   |  **pt**  |       [CCAligned](https://opus.nlpl.eu/CCAligned.php)        |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 12   |  **it**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), [Wikipedia](https://opus.nlpl.eu/Wikipedia.php) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 13   |  **ja**  | [MultiParaCrawl](https://opus.nlpl.eu/MultiParaCrawl.php) ([Creative Commons CC0 license](https://creativecommons.org/share-your-work/public-domain/cc0/) ) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 14   |  **zh**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), [TSL2019](https://github.com/brightmart/nlp_chinese_corpus) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)), wudaoMM([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode))[1] |
| 15   |  **es**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 16   |  **de**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), [EUbookshop](https://opus.nlpl.eu/EUbookshop.php) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 17   |  **fr**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), EuroPat ([Creative Commons CC0 license](https://creativecommons.org/share-your-work/public-domain/cc0/)) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |
| 18   |  **ru**  | WikiMatrix ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), [CCMatrix](https://opus.nlpl.eu/CCMatrix.php) |      | LAION-Aesthetics ([MIT](https://github.com/LAION-AI/laion-datasets/blob/main/LICENSE)) |

\[1] WuDaoMM数据集仅用于学术研究，任何使用该数据集都应该遵循以下要求。WuDaoMM不拥有这些图片的版权。 图片的使用必须遵守[Flickr使用条款](http://creativecommons.org/licenses/by/4.0/)。 图像的用户对使用数据集承担全部责任，不私自传播上面的图片。 如果图片的版权受到侵犯，请联系我们，我们将立即删除。

[1] WuDaoMMdataset is only used for academic research, any use of this dataset should follow the following requirements. WuDaoMM does not own the copyright of these pictures. Use of images is subject to the [Flickr term of use](http://creativecommons.org/licenses/by/4.0/). Users of the images take full responsibility for using the dataset and do not distribute the above images privately. If the copyright of the image is violated, please contact us and it will be removed immediately.



阶段1使用平行语料库数据。

阶段2和3主要使用Laion-Aesthetics的一个子集。中文数据集采用wudaoMM数据集(CC-BY-SA 4.0)。

Stage 1 uses parallel corpus data. 

Stage2&3 mainly use a subset of Laion-Aesthetics. The wudaoMM data set (CC-BY-SA 4.0) is used as a Chinese data set.



## 引用 Citation

关于AltCLIP，我们已经推出了相关报告，有更多细节可以查阅，如对您的工作有帮助，欢迎引用。

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

## AltCLIP-m18评测  AltCLIP-m18 evaluation

部分数据集评测结果展示:

Partial dataset evaluation results are displayed:

|  | flickr30k I2T | flickr30k  T2I | flickr30k_cn I2T | flickr30k_cn T2I | image-r | image-r-cn | image-r-es | Image-r-fr | image-r-it | Image-r-jp | Image-r-ko | Image-r-ru | birdsnap | caltech101 | cars | cifar10 | cifar100 | country211 | dtd | eurosat | fer2013 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| AltCLIP-M18 | 91.1 | 77.76 | 87.9 | 74.46 | 89.53 | 81.3650.35 | 71.78 | 74.96 | 76.44 | 67.68 | 69.27 | 75.53 | 41.57 | 88.25 | 92.75 | 97.44 | 84.83 | 30.52 | 68.62 | 67.46 | 54.4 |

Cifar10 dataset evaluation

```python
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

dataset_root = "./clip_benchmark_datasets/"
dataset_name = "cifar10"

auto_loader = AutoLoader(
    task_name="txt_img_matching",
    model_dir="./checkpoints/",
    model_name="AltCLIP-XLMR-L-m18"   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
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

```



##  推理脚本 inference

```python
import torch
from PIL import Image
from flagai.auto_model.auto_loader import AutoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(
    task_name="txt_img_matching",
    model_name="AltCLIP-XLMR-L-m18",   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
    model_dir="./checkpoints"
)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
transform = loader.get_transform()

model.eval()
model.to(device)
tokenizer = loader.get_tokenizer()

def inference():
    image = Image.open("./dog.jpeg")
    image = transform(image)
    image = torch.tensor(image["pixel_values"]).to(device)
    tokenizer_out = tokenizer(["a rat", "a dog", "a cat"], 
                                padding=True,
                                truncation=True,
                                max_length=77,
                                return_tensors='pt')

    text = tokenizer_out["input_ids"].to(device)
    attention_mask = tokenizer_out["attention_mask"].to(device)
    with torch.no_grad():
        image_features = model.get_image_features(image)
        text_features = model.get_text_features(text, attention_mask=attention_mask)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)

    print(text_probs.cpu().numpy()[0].tolist())

if __name__=="__main__":
    inference()
```



## 微调 fintuning 

Cifar10 dataset 

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
    model_dir="./checkpoints",
    model_name="AltCLIP-XLMR-L-m18"   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
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
    input_ids = torch.tensor([tokenizer(f"a photo of a {b[1]}",
                                padding=True,
                                truncation=True,
                                max_length=77)["input_ids"] for b in batch])    

    attention_mask = torch.tensor([tokenizer(f"a photo of a {b[1]}",
                                padding=True,
                                truncation=True,
                                max_length=77)["attention_mask"] for b in batch])

    return {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
if __name__ == "__main__":
    trainer.train(model=model, train_dataset=dataset, collate_fn=cifar10_collate_fn)
```

