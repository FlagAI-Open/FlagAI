# Contrastive Language-Image Pre-Training with [EVA](https://github.com/baaivision/EVA) (EVA-CLIP)

## Model Card

| model name | #param. | precision | data  |  batch size | IN-1K zero-shot top-1 |
|:-----------:|:------:|:------:|:------:|:------:|:------:|
| `eva-clip` | 1.3B | `fp16` | [LAION-400M](https://laion.ai/laion-400-open-dataset/) | 41K | 78.5


To our knowledge, EVA-CLIP is the largest performant open-sourced CLIP model evaluated via zero-shot classification performance.

For more details of EVA-CLIP, please refer to Section 2.3.5 of [paper](https://arxiv.org/pdf/2211.07636.pdf).

## Performance

| dataset | acc1 | acc5 | mean_per_class_recall  | 
|:-----------:|:------:|:------:|:------:|
| `imagenet1k` | 78.53 | 95.51 | 78.51 |
| `imagenet-a` | 73.59 | 90.93 | 69.97 |
| `imagenet-r` | 92.5 | 98.24 | 91.19 |
| `imagenet-sketch` | 67.31 | 89.07 | 67.31 |
| `imagenetv2` | 71.52 | 92.11 | 71.56 |
| `objectnet` | 72.33 | 89.37 | 70.88 |

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

def inference():
    image = Image.open("./CLIP.png")
    image = transform(image).unsqueeze(0).to(device)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)

    print(text_probs.cpu().numpy()[0].tolist())
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
text_inputs = torch.cat([tokenizer.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

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