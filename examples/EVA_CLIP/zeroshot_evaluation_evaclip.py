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