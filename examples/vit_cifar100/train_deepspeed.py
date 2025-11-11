"""
DeepSpeed Training Script for ViT on CIFAR-100
Compatible with DeepSpeed latest version and best practices

Usage:
    # Single node, multiple GPUs
    deepspeed --num_gpus=8 train_deepspeed.py
    
    # Or using torchrun (recommended)
    torchrun --nproc_per_node=8 train_deepspeed.py
    
    # Multi-node
    deepspeed --num_gpus=8 --num_nodes=2 --hostfile=./hostfile train_deepspeed.py
    # Or using torchrun
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 train_deepspeed.py
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 train_deepspeed.py

DeepSpeed Configuration:
    The script uses deepspeed.json for DeepSpeed configuration.
    Make sure the config file is compatible with the latest DeepSpeed version.
    For latest DeepSpeed features, refer to: https://www.deepspeed.ai/docs/config-json/
"""
import torch
import os
from torchvision import transforms
from torchvision.datasets import CIFAR100
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader

# Training hyperparameters
lr = 2e-5
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get distributed environment variables (set by deepspeed/torchrun)
# These are automatically set when using deepspeed or torchrun
local_rank = int(os.environ.get('LOCAL_RANK', 0))
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

# DeepSpeed configuration
env_type = "deepspeed"
num_gpus = int(os.environ.get('WORLD_SIZE', 8))  # Use WORLD_SIZE from launcher if available
deepspeed_config_path = './deepspeed.json'

# Validate DeepSpeed config file exists
if not os.path.exists(deepspeed_config_path):
    print(f"Warning: DeepSpeed config file {deepspeed_config_path} not found.")
    print("Please create a DeepSpeed configuration file. See examples for reference.")
    # You can create a default config here if needed

trainer = Trainer(
    env_type=env_type,
    experiment_name="vit-cifar100-deepspeed",
    batch_size=150,  # Per-GPU batch size (will be overridden by deepspeed.json if specified)
    num_gpus=num_gpus,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=1,  # Will be overridden by deepspeed.json if specified
    lr=lr,  # Will be overridden by deepspeed.json if optimizer is specified there
    weight_decay=1e-5,
    epochs=n_epochs,
    log_interval=100,
    eval_interval=1000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_vit_cifar100_deepspeed",
    save_interval=1000,
    num_checkpoints=1,  # Number of activation checkpoints
    hostfile="./hostfile",  # Optional: only needed for multi-node
    deepspeed_config=deepspeed_config_path,
    training_script="train_deepspeed.py",
    # DeepSpeed specific settings
    deepspeed_activation_checkpointing=True,  # Enable activation checkpointing for memory efficiency
)

def build_cifar():
    """
    Build CIFAR-100 datasets with data augmentation.
    Compatible with PyTorch 2.0+ transforms and DeepSpeed.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR100(root="./data/cifar100", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root="./data/cifar100", train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def collate_fn(batch):
    """
    Collate function for batching data.
    Compatible with DeepSpeed and supports mixed precision training.
    """
    images = torch.stack([b[0] for b in batch])
    # DeepSpeed handles mixed precision automatically, but we can pre-convert if needed
    if hasattr(trainer, 'fp16') and trainer.fp16:
        images = images.half()
    labels = [b[1] for b in batch]
    labels = torch.tensor(labels, dtype=torch.long)  # Use dtype instead of .long()
    return {"images": images, "labels": labels}

def validate(logits, labels, meta=None):
    """
    Validation metric function for accuracy calculation.
    Compatible with DeepSpeed and supports distributed evaluation.
    """
    _, predicted = logits.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    """
    Main training loop with DeepSpeed.
    Compatible with latest DeepSpeed version and best practices.
    
    Note:
    - DeepSpeed will automatically handle optimizer and scheduler if specified in config
    - If optimizer/scheduler are specified in deepspeed.json, they will override the ones passed here
    - The Trainer class handles all DeepSpeed initialization automatically
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load model
    loader = AutoLoader(task_name="classification",
                        model_name="vit-base-p16-224",
                        num_classes=100)

    model = loader.get_model()
    
    # Setup optimizer and scheduler
    # Note: If optimizer is specified in deepspeed.json, it will override this
    # DeepSpeed will automatically wrap the optimizer with ZeRO if configured
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-5,
        # PyTorch 2.0+ fused optimizer (if available)
        fused=torch.cuda.is_available() and hasattr(torch.optim.Adam, 'fused')
    )
    
    # Learning rate scheduler
    # Note: If scheduler is specified in deepspeed.json, it will override this
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    # Load datasets
    train_dataset, val_dataset = build_cifar()

    # Start training with DeepSpeed
    # The Trainer class handles all DeepSpeed setup automatically:
    # - Initializes DeepSpeed engine
    # - Handles ZeRO optimization
    # - Manages mixed precision training
    # - Handles gradient accumulation
    # - Manages checkpointing
    trainer.train(
        model,
        optimizer=optimizer,  # May be overridden by deepspeed.json
        lr_scheduler=scheduler,  # May be overridden by deepspeed.json
                  train_dataset=train_dataset,
                  valid_dataset=val_dataset,
                  metric_methods=[["accuracy", validate]],
        collate_fn=collate_fn,
        find_unused_parameters=False  # Set to False for better performance if all parameters are used
    )
    
    # Cleanup (handled automatically by Trainer and DeepSpeed, but good practice)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()





