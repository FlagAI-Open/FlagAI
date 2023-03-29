# Contrastive Language-Image Pre-Training with [EVA](https://github.com/baaivision/EVA) (EVA-CLIP)

## Model Card

| model name | #param. | precision | data  |  batch size | IN-1K zero-shot top-1 | Weights |
|:-----------:|:------:|:------:|:------:|:------:|:------:|:------:|
| `eva-clip` | 1.3B | `fp16` | [LAION-400M](https://laion.ai/laion-400-open-dataset/) | 41K | 78.5 | [ModelHub Link](https://model.baai.ac.cn/model-detail/100080) |


To our knowledge, EVA-CLIP is the largest performant open-sourced CLIP model evaluated via zero-shot classification performance.

For more details of EVA-CLIP, please refer to Section 2.3.5 of [paper](https://arxiv.org/pdf/2211.07636.pdf).

## EVA-CLIP Zero-shot Evaluation Results

### Zero-shot Image Classification Evaluation

The top-1 accuracy of ImageNet-1K variants and ObjectNet.

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with)](https://paperswithcode.com/sota/self-supervised-image-classification-with?p=eva-exploring-the-limits-of-masked-visual) 

| model | IN-1K | IN-V2 |  IN-Adv. | IN-Ren. |IN-Ske. | ObjectNet |
|-------|:-----:|:-----:|:----:| :------:|:-------:|:---------:|
| OpenAI CLIP-L | 75.55 | 69.86 | 70.76 | 87.83 | 59.58 | 68.98 |
| Open CLIP-H | 77.96 | 70.87 | 59.33 | 89.33 | 66.58 | 69.71 |
| Open CLIP-g | 76.65 | 69.56 | 57.19 | 88.69 | 65.17 | 67.53 |
| EVA CLIP-g | **78.53** | **71.52** | **73.59** | **92.5** | **67.31** | **72.33** |
 
</div>

### Zero-shot Video Action Recognition Evaluation

The performance of video action recognition benchmarks.

<div align="center">

| model | UCF-101 | Kinetics-400 | Kinetics-600 | Kinetics-700 |
|-------|:-----:|:-----:|:----:| :----:|
| OpenAI CLIP-L | 76.39 | 64.47 | 64.21 | 57.68 |
| Open CLIP-H   | **78.16** | 63.06 | 63.58 | 56.09 |
| Open CLIP-g   | 77.73 | 61.69 | 62.16 | 54.99 |
| EVA CLIP-g    | 76.05 | **65.23** | **64.38** | **58.4** |

</div>


> For video action recognition, we sample only a single center frame each video, turning it into an image classification task.
> Following the conventional settings, we report the top-1 accuracy for UCF-101 and the mean of top-1 and top-5 accuracy for Kinetics-400/600/700.

### Zero-shot Retrieval Evaluation

<div align="center">

<table>
   <tr>
      <td rowspan=2>Dataset</td>
      <td rowspan=2>Model</td>
      <td colspan=3>Text-to-Image Retrival</td>
      <td colspan=3>Image-to-Text Retrival</td>
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
      <td rowspan=4>Flickr30k</td>
      <td>OpenAI CLIP-L</td>
      <td>65.18 </td>
      <td>87.28 </td>
      <td>92 </td>
      <td>85.2 </td>
      <td>97.3 </td>
      <td>99 </td>
   </tr>
   <tr>
      <td>Open CLIP-H</td>
      <td><b>77.78</b></td>
      <td><b>94.14</b></td>
      <td><b>96.62</b></td>
      <td><b>90.8</b></td>
      <td><b>99.3</b></td>
      <td>99.7</td>
   </tr>
   <tr>
      <td>Open CLIP-g</td>
      <td>76.52 </td>
      <td>93.62 </td>
      <td>96.28 </td>
      <td>90.8 </td>
      <td>99.1 </td>
      <td><b>99.8</b></td>
   </tr>
   <tr>
      <td>EVA CLIP-g</td>
      <td>72.64 </td>
      <td>91.6 </td>
      <td>95.12 </td>
      <td>88.3 </td>
      <td>98.3 </td>
      <td>99.3 </td>
   </tr>
   <tr>
      <td rowspan=4>MSCOCO</td>
      <td>OpenAI CLIP-L</td>
      <td>36.51 </td>
      <td>61.01 </td>
      <td>71.11 </td>
      <td>56.34 </td>
      <td>79.32 </td>
      <td>86.66 </td>
   </tr>
   <tr>
      <td>Open CLIP-H</td>
      <td><b>49.47</b></td>
      <td><b>73.4</b></td>
      <td><b>81.53</b></td>
      <td><b>65.96</b></td>
      <td><b>86.06</b></td>
      <td><b>91.9</b></td>
   </tr>
   <tr>
      <td>Open CLIP-g</td>
      <td>47.99 </td>
      <td>72.37 </td>
      <td>80.75 </td>
      <td>64.96 </td>
      <td>85.3 </td>
      <td>91.46 </td>
   </tr>
   <tr>
      <td>EVA CLIP-g</td>
      <td>44.07 </td>
      <td>68.5 </td>
      <td>77.33 </td>
      <td>61.76 </td>
      <td>83.28 </td>
      <td>89.96 </td>
   </tr>
</table>

</div>

> The zero-shot retrieval performance of EVA-CLIP is relatively inferior to the Open CLIP-H / -g counterpart. We speculate there are two main reasons: 
> - The size / capacity of the language tower in EVA-CLIP is much smaller / weaker than Open CLIP-H and Open CLIP-g, *i.e.*, `124M` *v.s.* `354M`, and is only `~1/8` of the vision tower. Meanwhile, retrieval tasks depend more on the capacity of the language branch compared with classification tasks.
> - Retrieval tasks seem benefit more from the training dataset size (LAION-2B used by Open CLIP), while we only leverage LAION-400M for EVA-CLIP training. 
> Nevertheless, it is hard to make a head-to-head comparison between different CLIP models. In the future, we will further scale up the language encoder & training data to improve the retrieval performance.

## Usage

```python
import torch
from PIL import Image
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.dataset.mm.clip_dataset import clip_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="txt_img_matching", #contrastive learning
                    model_name="eva-clip")

model = loader.get_model()
model.eval()
model.to(device)
tokenizer = loader.get_tokenizer()
transform = clip_transform(img_size=model.visual.image_size)

def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream

def inference():
    # local image
    # image = Image.open(/path/to/image)
    # online image
    image = Image.open(download_image("https://bkimg.cdn.bcebos.com/pic/4610b912c8fcc3ce2d02315d9d45d688d53f209a?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxMTY=,g_7,xp_5,yp_5"))
    image = transform(image).unsqueeze(0).to(device)
    text = tokenizer.tokenize_as_tensor(["a tomato", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)

    print(text_probs.cpu().numpy()[0].tolist()) # [1.0, 0.0]
```

## Zero-Shot Prediction
The code below performs zero-shot prediction using EVA_CLIP. This example takes an image from the CIFAR-100 dataset, and predicts the most likely labels among the 100 textual labels from the dataset.

```python
import os
import torch
from torchvision.datasets import CIFAR100
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.dataset.mm.clip_dataset import clip_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="txt_img_matching", #contrastive learning
                    model_name="eva-clip")

model = loader.get_model()
model.eval()
model.to(device)
tokenizer = loader.get_tokenizer()
transform = clip_transform(img_size=model.visual.image_size)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = transform(image).unsqueeze(0).to(device)
text_inputs = torch.cat([tokenizer.tokenize_as_tensor(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

```
The output will look like the following (the exact numbers may be slightly different depending on the compute device):
```bash
Top predictions:

           snake: 100.00%
          turtle: 0.00%
     caterpillar: 0.00%
            worm: 0.00%
         leopard: 0.00%
```

## Acknowledgement

EVA-CLIP is built with [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip) and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark).
Thanks for their awesome works!
