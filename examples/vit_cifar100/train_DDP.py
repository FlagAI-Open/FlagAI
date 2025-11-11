"""
PyTorch DDP Training Script for ViT on CIFAR-100
Compatible with PyTorch 2.0+ and latest DDP best practices

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=8 train_DDP.py
    
    # Multi-node (using hostfile)
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 train_DDP.py
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 train_DDP.py
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

# Get distributed environment variables (set by torchrun)
# These are automatically set when using torchrun
local_rank = int(os.environ.get('LOCAL_RANK', 0))
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

# DDP configuration
env_type = "pytorchDDP"
num_gpus = int(os.environ.get('WORLD_SIZE', 8))  # Use WORLD_SIZE from torchrun if available

trainer = Trainer(
    env_type=env_type,
    experiment_name="vit-cifar100-8gpu",
    batch_size=150,  # Per-GPU batch size
    num_gpus=num_gpus,
    gradient_accumulation_steps=1,
    lr=lr,
    weight_decay=1e-5,
    epochs=n_epochs,
    log_interval=100,
    eval_interval=1000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_vit_cifar100_8gpu",
    save_interval=1000,
    num_checkpoints=1,
    hostfile="./hostfile",  # Optional: only needed for multi-node
    training_script="train_DDP.py",
    # PyTorch 2.0+ optimizations
    # Note: find_unused_parameters is handled by Trainer class
    # Set to False for better performance if all parameters are used
)

def build_cifar():
    """
    Build CIFAR-100 datasets with data augmentation.
    Compatible with PyTorch 2.0+ transforms.
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
    Compatible with PyTorch 2.0+ and supports mixed precision.
    """
    images = torch.stack([b[0] for b in batch])
    # Move to device if needed (handled by Trainer, but kept for compatibility)
    if hasattr(trainer, 'fp16') and trainer.fp16:
        images = images.half()
    labels = [b[1] for b in batch]
    labels = torch.tensor(labels, dtype=torch.long)  # Use dtype instead of .long()
    return {"images": images, "labels": labels}

def validate(logits, labels, meta=None):
    """
    Validation metric function for accuracy calculation.
    Compatible with PyTorch 2.0+ and supports distributed evaluation.
    """
    _, predicted = logits.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    """
    Main training loop.
    Compatible with PyTorch 2.0+ DDP best practices.
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
    
    # Setup optimizer with PyTorch 2.0+ optimizations
    # Using fused=True for better performance (requires PyTorch 2.0+)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-5,
        fused=torch.cuda.is_available() and hasattr(torch.optim.Adam, 'fused')  # PyTorch 2.0+ feature
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    # Load datasets
    train_dataset, val_dataset = build_cifar()

    # Start training with DDP
    # The Trainer class handles all DDP setup automatically
    trainer.train(
        model,
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  train_dataset=train_dataset,
                  valid_dataset=val_dataset,
                  metric_methods=[["accuracy", validate]],
        collate_fn=collate_fn,
        find_unused_parameters=False  # Set to False for better performance if all parameters are used
    )
    
    # Cleanup (handled automatically by Trainer, but good practice)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()





