import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def clip_transform(img_size):
    transform = Compose([
    Resize(img_size, interpolation=Image.BICUBIC),
    CenterCrop(img_size),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
  
    return transform

class CsvDataset(Dataset):
    def __init__(self, 
                input_filename,
                img_dir,
                transforms, 
                tokenizer, 
                img_key="filepath", 
                caption_key="title", 
                sep="\t"):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.img_dir = img_dir
        self.img_names = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.tokenizer = tokenizer
        print('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.img_names[idx]))
        images = self.transforms(image)
        texts = self.tokenizer.tokenize_as_tensor([str(self.captions[idx])])[0]
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