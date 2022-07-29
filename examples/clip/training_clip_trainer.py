import sys
import torch
from PIL import Image
from flagai.model.clip_model import CLIP
from flagai.data.transform import image_transform #文件位置待确定
from flagai.data.tokenizer.clip import tokenizer
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from flagai.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir = "/mnt/clip_models/ViT-B-32"
data_path = "/mnt/datasets/multimodal/ConceptualCaptions/Train_GCC-training_output.csv"
image_path = "/mnt/datasets/multimodal/ConceptualCaptions/"

trainer = Trainer(env_type="pytorch",
                  epochs=5,
                  pytorch_device=device,
                  batch_size=4,
                  lr=1e-4,
                  log_interval=10,
                  eval_interval=100,
                  warm_up=0
                  )

def get_image_transform(image_size, mean=None, std=None):
    preprocess_train = image_transform(image_size, is_train=True, mean=mean, std=std)
    preprocess_val = image_transform(image_size, is_train=False, mean=mean, std=std)
    return preprocess_train, preprocess_val

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
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

def collate_fn(batch):
    images = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    images = torch.stack(images, dim=0).float()
    texts = torch.stack(texts, dim=0).long()
    return {
        "image": images,
        "text": texts
    }

model = CLIP.init_from_json(os.path.join(dir,"config.json")).to(device)

train_transform, val_transform = get_image_transform(224)
dataset = CsvDataset(data_path, train_transform, "filepath", "title")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

trainer.train(model, train_dataset=dataset, collate_fn=collate_fn, optimizer=optimizer)



