import torch
from flagai.data.dataset.mm.clip_dataset import CsvDataset, clip_transform, collate_fn
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cd examples/clip
data_path = "./data/pairs.csv"#"/mnt/datasets/multimodal/ConceptualCaptions/Train_GCC-training_output.csv"
img_dir = "./data/img"#"/mnt/datasets/multimodal/ConceptualCaptions"

trainer = Trainer(
    env_type="deepspeed",
    experiment_name="clip",
    batch_size=64,
    num_gpus=2,
    fp16=True,
    gradient_accumulation_steps=1,
    lr=1e-4,
    weight_decay=1e-5,
    epochs=5,
    log_interval=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="clip_deepspeed",
    save_interval=1000,
    num_checkpoints=1,
    hostfile="./deepspeed/hostfile",
    training_script=__file__,
    deepspeed_config="./deepspeed/deepspeed.json"
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

