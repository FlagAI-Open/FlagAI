import torch
from flagai.data.dataset.mm.clip_dataset import CsvDataset, clip_transform, collate_fn
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader

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
                  )

loader = AutoLoader(task_name="txt_img_matching",#contrastive learning
                    model_name="clip-base-p32-224",
                    )
model = loader.get_model()
tokenizer = loader.get_tokenizer()

transform = clip_transform(img_size=model.image_size)
train_dataset = CsvDataset(data_path,
                            img_dir,
                            transform,
                            tokenizer)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
trainer.train(model,
              optimizer=optimizer,
              train_dataset=train_dataset,
              collate_fn=collate_fn)

