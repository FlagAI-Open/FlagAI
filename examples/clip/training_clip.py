import sys
import torch
from PIL import Image
from flagai.model.clip_model import CLIP
from flagai.data.transform import image_transform #文件位置待确定
from flagai.data.tokenizer.clip import tokenizer
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir = "/mnt/clip_models/ViT-B-32"
data_path = "/mnt/datasets/multimodal/ConceptualCaptions/Train_GCC-training_output.csv"
image_path = "/mnt/datasets/multimodal/ConceptualCaptions/"

def get_image_transform(image_size, mean=None, std=None):
    preprocess_train = image_transform(image_size, is_train=True, mean=mean, std=std)
    preprocess_val = image_transform(image_size, is_train=False, mean=mean, std=std)
    return preprocess_train, preprocess_val

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        for i, row in df.iterrows():
            print(row)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        print('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(os.path.join(image_path, str(self.images[idx]))))
        texts = tokenizer.tokenize([str(self.captions[idx])])[0]
        return images, texts


model = CLIP.init_from_json(os.path.join(dir,"config.json")).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

train_transform, val_transform = get_image_transform(224)
dataset = CsvDataset(data_path, train_transform, "filepath", "title")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True,)

for image, text in dataloader:
    print(image.shape)
    print(text.shape)
    image = image.to(device)
    text = text.to(device)
    optimizer.zero_grad()
    model_out = model(**{"image": image, "text": text})
    loss = model_out["loss"]
    print(f"loss is {loss}")
    loss.backward()
    optimizer.step()



