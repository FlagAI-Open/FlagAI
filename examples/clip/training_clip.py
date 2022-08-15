import sys
sys.path.append("/mnt/wchh/FlagAI-internal")
import torch
from PIL import Image
from flagai.data.tokenizer.clip import tokenizer
import os
from torch.utils.data import Dataset
import pandas as pd
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cd examples/clip
data_path = "./data/pairs.csv"
img_dir = "./data/img"

trainer = Trainer(env_type="pytorch",
                  epochs=5,
                  pytorch_device=device,
                  batch_size=64,
                  lr=1e-4,
                  log_interval=10,
                  eval_interval=100,
                  )

# trainer = Trainer(env_type="pytorchDDP",
#                   epochs=5,
#                   pytorch_device=device,
#                   batch_size=4,
#                   lr=1e-4,
#                   log_interval=10,
#                   eval_interval=100,
#                   num_gpus=2,
#                   )
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def build_dataset(n_px):
    transform_train = Compose([
    Resize(n_px, interpolation=Image.BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
    # only training
    train_dataset = CsvDataset(data_path, transform_train, "filepath", "title")

    return train_dataset

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        print(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.img_names = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        print('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(img_dir, str(self.img_names[idx])))
        images = self.transforms(image)
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

loader = AutoLoader(task_name="cl",#contrastive learning
                    model_name="clip-base-p32-224",
                    model_dir="/mnt/clip_models/")
model = loader.get_model()
tokenizer = loader.get_tokenizer()

train_dataset = build_dataset(model.image_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
trainer.train(model,
              optimizer=optimizer,
              train_dataset=train_dataset,
              collate_fn=collate_fn)

